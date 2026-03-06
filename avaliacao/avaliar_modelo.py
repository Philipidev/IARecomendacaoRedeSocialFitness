from __future__ import annotations

import argparse
import ast
import json
import pickle
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

ROOT = Path(__file__).resolve().parent.parent
MODELO_DIR = ROOT / "treinamento" / "modelo"
SPLITS_DIR = ROOT / "treinamento" / "dados" / "splits"
OUTPUT_DIR = ROOT / "extracao_filtragem" / "output"
RESULTADOS_DIR = ROOT / "avaliacao" / "resultados"

PESO_COSINE = 0.40
PESO_COOC = 0.25
PESO_TIME = 0.15
PESO_SOCIAL = 0.20
LAMBDA_DECAY = 0.01
MS_POR_DIA = 86_400_000

K_PADRAO = [5, 10, 20]


def _parse_tags(value) -> list[str]:
    if isinstance(value, (list, np.ndarray)):
        return [str(t) for t in value]
    if isinstance(value, str):
        try:
            parsed = ast.literal_eval(value)
            return [str(t) for t in parsed] if isinstance(parsed, list) else [value]
        except Exception:
            return [value]
    return []


def _normalizar_k(lista_k: Iterable[int]) -> list[int]:
    ks = sorted({int(k) for k in lista_k if int(k) > 0})
    if not ks:
        raise ValueError("Informe pelo menos um K positivo.")
    return ks


def _detectar_coluna_tempo(interactions: pd.DataFrame) -> str | None:
    candidatas = [
        "event_timestamp",
        "event_time",
        "timestamp",
        "created_at",
        "interaction_date",
        "creation_date",
    ]
    for col in candidatas:
        if col in interactions.columns:
            return col
    return None


@dataclass
class ArtefatosModelo:
    vectorizer: object
    post_matrix: np.ndarray
    cooccurrence_map: dict[str, list[tuple[str, float]]]
    social_scores: np.ndarray
    posts_cache: pd.DataFrame


def carregar_artefatos_modelo() -> ArtefatosModelo:
    obrigatorios = [
        "vectorizer.pkl",
        "post_matrix.npy",
        "tag_cooccurrence_map.pkl",
        "posts_cache.parquet",
    ]
    faltantes = [nome for nome in obrigatorios if not (MODELO_DIR / nome).exists()]
    if faltantes:
        raise FileNotFoundError(
            "Artefatos ausentes em treinamento/modelo/: "
            + ", ".join(faltantes)
            + "\nExecute: python treinamento/treinar.py"
        )

    with open(MODELO_DIR / "vectorizer.pkl", "rb") as f:
        vectorizer = pickle.load(f)

    post_matrix = np.load(MODELO_DIR / "post_matrix.npy")

    with open(MODELO_DIR / "tag_cooccurrence_map.pkl", "rb") as f:
        cooccurrence_map = pickle.load(f)

    social_path = MODELO_DIR / "social_scores.npy"
    social_scores = np.load(social_path) if social_path.exists() else np.zeros(post_matrix.shape[0], dtype=np.float32)

    posts_cache = pd.read_parquet(MODELO_DIR / "posts_cache.parquet")
    posts_cache["tags_fitness"] = posts_cache["tags_fitness"].apply(_parse_tags)

    if len(posts_cache) != post_matrix.shape[0]:
        raise ValueError(
            "Inconsistência entre posts_cache.parquet e post_matrix.npy "
            f"({len(posts_cache)} vs {post_matrix.shape[0]})."
        )

    return ArtefatosModelo(
        vectorizer=vectorizer,
        post_matrix=post_matrix,
        cooccurrence_map=cooccurrence_map,
        social_scores=social_scores,
        posts_cache=posts_cache,
    )


def carregar_splits_teste() -> tuple[pd.DataFrame, pd.DataFrame]:
    test_posts_path = SPLITS_DIR / "test_posts.parquet"
    test_inter_path = SPLITS_DIR / "test_interactions.parquet"
    if not test_posts_path.exists() or not test_inter_path.exists():
        raise FileNotFoundError(
            "Splits de teste não encontrados em treinamento/dados/splits/. "
            "Execute: python treinamento/dividir_dataset.py"
        )

    test_posts = pd.read_parquet(test_posts_path)
    test_posts["tags_fitness"] = test_posts["tags_fitness"].apply(_parse_tags)
    test_interactions = pd.read_parquet(test_inter_path)
    return test_posts, test_interactions


def construir_mapa_message_id() -> dict[int, int]:
    msgs_path = OUTPUT_DIR / "messages_fitness.parquet"
    if not msgs_path.exists():
        return {}

    msgs_df = pd.read_parquet(msgs_path)
    if "message_id" not in msgs_df.columns:
        return {}

    return {idx: int(mid) for idx, mid in enumerate(msgs_df["message_id"].tolist())}


def _score_cosine(vectorizer, post_matrix: np.ndarray, tags: list[str]) -> np.ndarray:
    x = vectorizer.transform([tags]).astype(np.float32)
    if x.sum() == 0:
        return np.zeros(post_matrix.shape[0], dtype=np.float32)
    return cosine_similarity(x, post_matrix).flatten().astype(np.float32)


def _score_cooccurrence(
    cooccurrence_map: dict[str, list[tuple[str, float]]],
    classes_conhecidas: set[str],
    posts_tags: pd.Series,
    tags_entrada: list[str],
) -> np.ndarray:
    boost_por_tag: dict[str, float] = {}
    for tag in tags_entrada:
        for tag_vizinha, peso in cooccurrence_map.get(tag, []):
            if tag_vizinha in classes_conhecidas:
                boost_por_tag[tag_vizinha] = boost_por_tag.get(tag_vizinha, 0.0) + float(peso)

    if not boost_por_tag:
        return np.zeros(len(posts_tags), dtype=np.float32)

    scores = np.zeros(len(posts_tags), dtype=np.float32)
    for i, tags_post in enumerate(posts_tags):
        for tag in tags_post:
            scores[i] += boost_por_tag.get(tag, 0.0)

    max_score = float(scores.max())
    if max_score > 0:
        scores /= max_score
    return scores


def _score_time_decay(posts_cache: pd.DataFrame, timestamp_entrada: int) -> np.ndarray:
    if "creation_date" not in posts_cache.columns:
        return np.ones(len(posts_cache), dtype=np.float32)
    tempos = posts_cache["creation_date"].values.astype(np.float64)
    delta = np.abs(tempos - float(timestamp_entrada)) / MS_POR_DIA
    return np.exp(-LAMBDA_DECAY * delta).astype(np.float32)


def recomendar_ids(
    artefatos: ArtefatosModelo,
    tags_referencia: list[str],
    timestamp_referencia: int,
    top_k: int,
) -> tuple[list[int], list[list[str]], list[float], list[int]]:
    posts = artefatos.posts_cache
    tags_norm = [t.strip() for t in tags_referencia if str(t).strip()]
    if not tags_norm:
        return [], [], [], []

    sc = _score_cosine(artefatos.vectorizer, artefatos.post_matrix, tags_norm)
    si = _score_cooccurrence(
        artefatos.cooccurrence_map,
        set(artefatos.vectorizer.classes_),
        posts["tags_fitness"],
        tags_norm,
    )
    st = _score_time_decay(posts, timestamp_referencia)
    ss = artefatos.social_scores if len(artefatos.social_scores) == len(posts) else np.zeros(len(posts), dtype=np.float32)

    score_final = PESO_COSINE * sc + PESO_COOC * si + PESO_TIME * st + PESO_SOCIAL * ss
    ordem = np.argsort(-score_final)

    tags_set_ref = set(tags_norm)
    ids, tags, scores, idxs = [], [], [], []
    for i in ordem:
        tags_post = posts.iloc[int(i)]["tags_fitness"]
        if set(tags_post) == tags_set_ref:
            continue
        ids.append(int(posts.iloc[int(i)].name))
        tags.append(list(tags_post))
        scores.append(float(score_final[int(i)]))
        idxs.append(int(i))
        if len(ids) >= top_k:
            break

    return ids, tags, scores, idxs


def precision_at_k(relevantes: set[int], recomendados: list[int], k: int) -> float:
    if k == 0:
        return 0.0
    rec_k = recomendados[:k]
    if not rec_k:
        return 0.0
    hits = sum(1 for r in rec_k if r in relevantes)
    return hits / k


def recall_at_k(relevantes: set[int], recomendados: list[int], k: int) -> float:
    if not relevantes:
        return 0.0
    rec_k = recomendados[:k]
    hits = sum(1 for r in rec_k if r in relevantes)
    return hits / len(relevantes)


def hitrate_at_k(relevantes: set[int], recomendados: list[int], k: int) -> float:
    rec_k = recomendados[:k]
    return 1.0 if any(r in relevantes for r in rec_k) else 0.0


def map_at_k(relevantes: set[int], recomendados: list[int], k: int) -> float:
    if not relevantes:
        return 0.0

    soma_prec = 0.0
    hits = 0
    for i, rec in enumerate(recomendados[:k], start=1):
        if rec in relevantes:
            hits += 1
            soma_prec += hits / i

    den = min(len(relevantes), k)
    return soma_prec / den if den > 0 else 0.0


def ndcg_at_k(relevantes: set[int], recomendados: list[int], k: int) -> float:
    gains = [1.0 if rec in relevantes else 0.0 for rec in recomendados[:k]]
    dcg = sum(g / np.log2(i + 2) for i, g in enumerate(gains))

    ideal = [1.0] * min(len(relevantes), k)
    idcg = sum(g / np.log2(i + 2) for i, g in enumerate(ideal))
    if idcg == 0:
        return 0.0
    return float(dcg / idcg)


def diversidade_intra_lista(tags_recomendadas: list[list[str]]) -> float:
    n = len(tags_recomendadas)
    if n < 2:
        return 0.0

    distancias = []
    for i in range(n):
        a = set(tags_recomendadas[i])
        for j in range(i + 1, n):
            b = set(tags_recomendadas[j])
            uniao = a | b
            if not uniao:
                continue
            jacc = len(a & b) / len(uniao)
            distancias.append(1.0 - jacc)

    return float(np.mean(distancias)) if distancias else 0.0


def _timestamp_em_ms(valor) -> int | None:
    if pd.isna(valor):
        return None
    if isinstance(valor, (int, np.integer)):
        return int(valor)
    if isinstance(valor, (float, np.floating)):
        return int(valor)

    texto = str(valor)
    try:
        dt = pd.to_datetime(texto, utc=True)
        return int(dt.value // 1_000_000)
    except Exception:
        return None


def avaliar(artefatos: ArtefatosModelo, ks: list[int]) -> tuple[dict, pd.DataFrame, pd.DataFrame]:
    _, test_interactions = carregar_splits_teste()
    mapa_idx_message = construir_mapa_message_id()

    if "message_id" not in test_interactions.columns or "user_id" not in test_interactions.columns:
        raise ValueError("test_interactions.parquet precisa conter as colunas user_id e message_id.")

    tempo_col = _detectar_coluna_tempo(test_interactions)
    if tempo_col is None:
        posts_cache = artefatos.posts_cache
        if "creation_date" in posts_cache.columns and posts_cache.index.name is not None:
            tempo_col = "__fallback_order"
            test_interactions = test_interactions.reset_index(drop=True)
            test_interactions[tempo_col] = np.arange(len(test_interactions))
        else:
            test_interactions = test_interactions.reset_index(drop=True)
            tempo_col = "__fallback_order"
            test_interactions[tempo_col] = np.arange(len(test_interactions))

    test_interactions = test_interactions.copy()
    test_interactions["__ts_ms"] = test_interactions[tempo_col].apply(_timestamp_em_ms)
    if test_interactions["__ts_ms"].isna().all():
        test_interactions["__ts_ms"] = np.arange(len(test_interactions), dtype=np.int64)

    catalogo_total = len(artefatos.posts_cache)
    uso_catalogo = set()
    ilads = []
    novidades = []
    recencias = []

    linhas_q = []
    acumuladores = {k: defaultdict(list) for k in ks}

    pop = Counter(test_interactions["message_id"].tolist())

    for user_id, grupo in test_interactions.groupby("user_id"):
        g = grupo.sort_values("__ts_ms")
        eventos = g[["message_id", "__ts_ms"]].to_dict("records")
        if len(eventos) < 2:
            continue

        for i, evento in enumerate(eventos[:-1]):
            mensagem_ref = int(evento["message_id"])
            ts_ref = int(evento["__ts_ms"])

            futuros = {int(x["message_id"]) for x in eventos[i + 1 :]}
            if not futuros:
                continue

            idx_ref = None
            for idx, mid in mapa_idx_message.items():
                if mid == mensagem_ref:
                    idx_ref = idx
                    break
            if idx_ref is None or idx_ref not in artefatos.posts_cache.index:
                continue

            post_ref = artefatos.posts_cache.loc[idx_ref]
            tags_ref = _parse_tags(post_ref.get("tags_fitness", []))
            timestamp_ref = int(post_ref.get("creation_date", ts_ref))

            rec_ids_idx, rec_tags, rec_scores, rec_local_idxs = recomendar_ids(
                artefatos,
                tags_referencia=tags_ref,
                timestamp_referencia=timestamp_ref,
                top_k=max(ks),
            )

            rec_ids_message = [mapa_idx_message.get(idx, -1) for idx in rec_ids_idx]
            rec_ids_message = [r for r in rec_ids_message if r != -1]

            if not rec_ids_message:
                continue

            uso_catalogo.update(rec_local_idxs)
            ilads.append(diversidade_intra_lista(rec_tags))

            for local_idx in rec_local_idxs:
                rec_msg = mapa_idx_message.get(int(artefatos.posts_cache.iloc[local_idx].name), None)
                if rec_msg is None:
                    continue
                freq = pop.get(rec_msg, 0)
                novidades.append(1.0 / np.log2(freq + 2.0))

                if "creation_date" in artefatos.posts_cache.columns:
                    rec_ts = int(artefatos.posts_cache.iloc[local_idx]["creation_date"])
                    recencias.append(abs(rec_ts - ts_ref) / MS_POR_DIA)

            for k in ks:
                acumuladores[k]["precision"].append(precision_at_k(futuros, rec_ids_message, k))
                acumuladores[k]["recall"].append(recall_at_k(futuros, rec_ids_message, k))
                acumuladores[k]["hitrate"].append(hitrate_at_k(futuros, rec_ids_message, k))
                acumuladores[k]["map"].append(map_at_k(futuros, rec_ids_message, k))
                acumuladores[k]["ndcg"].append(ndcg_at_k(futuros, rec_ids_message, k))

            linhas_q.append(
                {
                    "user_id": int(user_id),
                    "message_id_referencia": mensagem_ref,
                    "timestamp_referencia_ms": ts_ref,
                    "n_relevantes_futuros": len(futuros),
                    "n_recomendados": len(rec_ids_message),
                    "top1_recomendado": rec_ids_message[0] if rec_ids_message else None,
                    "top1_score": rec_scores[0] if rec_scores else None,
                }
            )

    rows_metricas = []
    resumo_metricas = {}

    for k in ks:
        metricas_k = {}
        for nome in ["precision", "recall", "hitrate", "map", "ndcg"]:
            valores = acumuladores[k][nome]
            media = float(np.mean(valores)) if valores else 0.0
            metricas_k[f"{nome}@{k}"] = media
            rows_metricas.append({"k": k, "metrica": nome, "valor": media})
        resumo_metricas.update(metricas_k)

    cobertura = len(uso_catalogo) / catalogo_total if catalogo_total > 0 else 0.0

    resumo_negocio = {
        "catalog_coverage": float(cobertura),
        "intra_list_diversity_tags": float(np.mean(ilads)) if ilads else 0.0,
        "novelty_inverse_popularity": float(np.mean(novidades)) if novidades else 0.0,
        "avg_recommended_recency_days": float(np.mean(recencias)) if recencias else 0.0,
    }

    metadata = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "n_queries_validas": len(linhas_q),
        "ks": ks,
        "protocolo": (
            "Para cada usuário no split de teste, ordena interações por tempo. "
            "Cada interação vira item de referência; o ground truth é o conjunto de interações futuras do mesmo usuário. "
            "O recomendador gera Top-K com base nas tags/timestamp do item de referência e comparamos com os itens futuros reais."
        ),
    }

    resumo = {
        "metadata": metadata,
        "ranking_metrics": resumo_metricas,
        "business_metrics": resumo_negocio,
    }

    return resumo, pd.DataFrame(rows_metricas), pd.DataFrame(linhas_q)


def salvar_resultados(resumo: dict, df_metricas: pd.DataFrame, df_queries: pd.DataFrame) -> tuple[Path, Path, Path]:
    RESULTADOS_DIR.mkdir(parents=True, exist_ok=True)

    caminho_json = RESULTADOS_DIR / "metricas_resumo.json"
    caminho_csv = RESULTADOS_DIR / "metricas_ranking_por_k.csv"
    caminho_md = RESULTADOS_DIR / "resumo_avaliacao.md"
    caminho_queries = RESULTADOS_DIR / "queries_avaliadas.csv"

    with open(caminho_json, "w", encoding="utf-8") as f:
        json.dump(resumo, f, ensure_ascii=False, indent=2)

    df_metricas.to_csv(caminho_csv, index=False)
    df_queries.to_csv(caminho_queries, index=False)

    linhas_md = [
        "# Resumo de Avaliação do Recomendador\n",
        f"Gerado em: `{resumo['metadata']['generated_at_utc']}`\n",
        f"Consultas válidas avaliadas: **{resumo['metadata']['n_queries_validas']}**\n",
        "## Protocolo de avaliação\n",
        resumo["metadata"]["protocolo"] + "\n",
        "## Métricas de ranking\n",
    ]

    for chave, valor in resumo["ranking_metrics"].items():
        linhas_md.append(f"- **{chave}**: {valor:.4f}\n")

    linhas_md.append("\n## Métricas de negócio/TCC\n")
    linhas_md.append(f"- **Cobertura de catálogo**: {resumo['business_metrics']['catalog_coverage']:.4f}\n")
    linhas_md.append(f"- **Diversidade intra-lista (tags)**: {resumo['business_metrics']['intra_list_diversity_tags']:.4f}\n")
    linhas_md.append(f"- **Novidade (popularidade inversa)**: {resumo['business_metrics']['novelty_inverse_popularity']:.4f}\n")
    linhas_md.append(f"- **Recência média recomendada (dias)**: {resumo['business_metrics']['avg_recommended_recency_days']:.4f}\n")

    with open(caminho_md, "w", encoding="utf-8") as f:
        f.writelines(linhas_md)

    return caminho_json, caminho_csv, caminho_md


def main() -> None:
    parser = argparse.ArgumentParser(description="Avaliação offline do modelo de recomendação.")
    parser.add_argument(
        "--k",
        nargs="+",
        default=K_PADRAO,
        type=int,
        help="Lista de K para métricas de ranking (ex: --k 5 10 20)",
    )
    args = parser.parse_args()

    ks = _normalizar_k(args.k)

    artefatos = carregar_artefatos_modelo()
    resumo, df_metricas, df_queries = avaliar(artefatos, ks)
    caminho_json, caminho_csv, caminho_md = salvar_resultados(resumo, df_metricas, df_queries)

    print("=== Avaliação concluída ===")
    print(f"Consultas avaliadas: {resumo['metadata']['n_queries_validas']}")
    print(f"Resultados JSON : {caminho_json}")
    print(f"Resultados CSV  : {caminho_csv}")
    print(f"Resumo Markdown : {caminho_md}")


if __name__ == "__main__":
    main()
