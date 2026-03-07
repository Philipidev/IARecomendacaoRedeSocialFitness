from __future__ import annotations

import argparse
import ast
import json
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from typing import Iterable

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from progress_utils import IterationProgress
from treinamento.model_utils import (
    infer_model_family,
    model_id_from_dir,
    rel_path,
    resolve_model_dir,
)
from treinamento.rankers import load_ranker

SPLITS_DIR = ROOT / "treinamento" / "dados" / "splits"
OUTPUT_DIR = ROOT / "extracao_filtragem" / "output"
RESULTADOS_DIR = ROOT / "avaliacao" / "resultados"
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
    for col in [
        "event_timestamp",
        "event_time",
        "timestamp",
        "created_at",
        "interaction_date",
        "creation_date",
    ]:
        if col in interactions.columns:
            return col
    return None


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


def construir_mapa_message_id_global() -> dict[int, int]:
    msgs_path = OUTPUT_DIR / "messages_fitness.parquet"
    if not msgs_path.exists():
        return {}
    msgs_df = pd.read_parquet(msgs_path)
    if "message_id" not in msgs_df.columns:
        return {}
    return {idx: int(mid) for idx, mid in enumerate(msgs_df["message_id"].tolist())}


def construir_lookup_catalogo(ranker) -> tuple[dict[int, int], dict[int, int]]:
    posts = ranker.artifacts.posts_cache
    message_to_row: dict[int, int] = {}
    if "_message_id" in posts.columns:
        values = pd.to_numeric(posts["_message_id"], errors="coerce")
        for row_pos, value in enumerate(values):
            if pd.notna(value):
                message_to_row[int(value)] = row_pos

    fallback_catalog = construir_mapa_message_id_global()
    return message_to_row, fallback_catalog


def recomendar_ids(
    ranker,
    tags_referencia: list[str],
    timestamp_referencia: int,
    top_k: int,
) -> tuple[list[int], list[list[str]], list[float], list[int]]:
    df = ranker.recommend_df(
        tags=tags_referencia,
        timestamp=timestamp_referencia,
        top_k=top_k,
        include_internal=True,
    )
    if df.empty:
        return [], [], [], []

    rec_ids: list[int] = []
    rec_tags: list[list[str]] = []
    rec_scores: list[float] = []
    rec_catalog_idxs: list[int] = []

    index_col = "index" if "index" in df.columns else None
    if index_col is None and ranker.artifacts.posts_cache.index.name in df.columns:
        index_col = ranker.artifacts.posts_cache.index.name

    for _, row in df.iterrows():
        if "_message_id" in row and pd.notna(row["_message_id"]):
            rec_ids.append(int(row["_message_id"]))
        else:
            continue

        rec_tags.append(_parse_tags(row.get("tags_fitness", [])))
        rec_scores.append(float(row.get("relevance_score", 0.0)))
        if "_catalog_index" in row and pd.notna(row["_catalog_index"]):
            rec_catalog_idxs.append(
                int(ranker.artifacts.posts_cache.index.get_loc(int(row["_catalog_index"])))
            )
        elif index_col is not None and pd.notna(row[index_col]):
            rec_catalog_idxs.append(
                int(ranker.artifacts.posts_cache.index.get_loc(int(row[index_col])))
            )
        else:
            rec_catalog_idxs.append(len(rec_catalog_idxs))

    return rec_ids, rec_tags, rec_scores, rec_catalog_idxs


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


def mrr_at_k(relevantes: set[int], recomendados: list[int], k: int) -> float:
    for i, rec in enumerate(recomendados[:k], start=1):
        if rec in relevantes:
            return 1.0 / i
    return 0.0


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


def _total_consultas_candidatas(interactions: pd.DataFrame) -> int:
    return int(
        sum(
            max(len(grupo) - 1, 0)
            for _, grupo in interactions.groupby("user_id")
        )
    )


def avaliar(
    ranker,
    ks: list[int],
    model_dir: Path,
) -> tuple[dict, pd.DataFrame, pd.DataFrame]:
    _, test_interactions = carregar_splits_teste()
    message_to_row, fallback_catalog = construir_lookup_catalogo(ranker)

    if "message_id" not in test_interactions.columns or "user_id" not in test_interactions.columns:
        raise ValueError("test_interactions.parquet precisa conter as colunas user_id e message_id.")

    tempo_col = _detectar_coluna_tempo(test_interactions)
    if tempo_col is None:
        posts_cache = ranker.artifacts.posts_cache
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

    catalogo_total = len(ranker.artifacts.posts_cache)
    uso_catalogo = set()
    ilads = []
    novidades = []
    recencias = []
    latencias_ms = []
    total_queries_candidatas = _total_consultas_candidatas(test_interactions)
    progress = IterationProgress(
        total=total_queries_candidatas,
        label=f"Avaliação {model_id_from_dir(model_dir)}",
        every_percent=5,
    )
    if total_queries_candidatas > 0:
        progress.start("Processando consultas candidatas")
    processed_queries = 0

    linhas_q = []
    acumuladores = {k: defaultdict(list) for k in ks}

    pop = Counter(test_interactions["message_id"].tolist())

    for user_id, grupo in test_interactions.groupby("user_id"):
        g = grupo.sort_values("__ts_ms")
        eventos = g[["message_id", "__ts_ms"]].to_dict("records")
        if len(eventos) < 2:
            continue

        for i, evento in enumerate(eventos[:-1]):
            processed_queries += 1
            try:
                mensagem_ref = int(evento["message_id"])
                ts_ref = int(evento["__ts_ms"])

                futuros = {int(x["message_id"]) for x in eventos[i + 1 :]}
                if not futuros:
                    continue

                if mensagem_ref in message_to_row:
                    row_pos_ref = message_to_row[mensagem_ref]
                    post_ref = ranker.artifacts.posts_cache.iloc[row_pos_ref]
                else:
                    idx_ref = None
                    for idx, mid in fallback_catalog.items():
                        if mid == mensagem_ref:
                            idx_ref = idx
                            break
                    if idx_ref is None or idx_ref not in ranker.artifacts.posts_cache.index:
                        continue
                    post_ref = ranker.artifacts.posts_cache.loc[idx_ref]

                tags_ref = _parse_tags(post_ref.get("tags_fitness", []))
                timestamp_ref = int(post_ref.get("creation_date", ts_ref))

                started = perf_counter()
                rec_ids_message, rec_tags, rec_scores, rec_local_idxs = recomendar_ids(
                    ranker,
                    tags_referencia=tags_ref,
                    timestamp_referencia=timestamp_ref,
                    top_k=max(ks),
                )
                latencias_ms.append((perf_counter() - started) * 1000.0)

                if not rec_ids_message:
                    continue

                uso_catalogo.update(rec_local_idxs)
                ilads.append(diversidade_intra_lista(rec_tags))

                for local_idx in rec_local_idxs:
                    rec_msg = None
                    if "_message_id" in ranker.artifacts.posts_cache.columns and local_idx < len(
                        ranker.artifacts.posts_cache
                    ):
                        maybe_mid = ranker.artifacts.posts_cache.iloc[local_idx].get("_message_id")
                        if pd.notna(maybe_mid):
                            rec_msg = int(maybe_mid)
                    if rec_msg is None:
                        continue
                    freq = pop.get(rec_msg, 0)
                    novidades.append(1.0 / np.log2(freq + 2.0))

                    if "creation_date" in ranker.artifacts.posts_cache.columns:
                        rec_ts = int(ranker.artifacts.posts_cache.iloc[local_idx]["creation_date"])
                        recencias.append(abs(rec_ts - ts_ref) / MS_POR_DIA)

                for k in ks:
                    acumuladores[k]["precision"].append(precision_at_k(futuros, rec_ids_message, k))
                    acumuladores[k]["recall"].append(recall_at_k(futuros, rec_ids_message, k))
                    acumuladores[k]["hitrate"].append(hitrate_at_k(futuros, rec_ids_message, k))
                    acumuladores[k]["map"].append(map_at_k(futuros, rec_ids_message, k))
                    acumuladores[k]["ndcg"].append(ndcg_at_k(futuros, rec_ids_message, k))
                    acumuladores[k]["mrr"].append(mrr_at_k(futuros, rec_ids_message, k))

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
            finally:
                if total_queries_candidatas > 0:
                    progress.log(processed_queries)

    rows_metricas = []
    resumo_metricas = {}

    for k in ks:
        metricas_k = {}
        for nome in ["precision", "recall", "hitrate", "map", "ndcg", "mrr"]:
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
        "model_dir": rel_path(model_dir),
        "model_id": model_id_from_dir(model_dir),
        "family": infer_model_family(model_dir),
        "n_queries_validas": len(linhas_q),
        "ks": ks,
        "latencia_inferencia_ms_p50": float(np.percentile(latencias_ms, 50)) if latencias_ms else 0.0,
        "latencia_inferencia_ms_p95": float(np.percentile(latencias_ms, 95)) if latencias_ms else 0.0,
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

    if total_queries_candidatas > 0:
        progress.finish(f"Consultas válidas: {len(linhas_q)}")

    return resumo, pd.DataFrame(rows_metricas), pd.DataFrame(linhas_q)


def salvar_resultados(
    resumo: dict,
    df_metricas: pd.DataFrame,
    df_queries: pd.DataFrame,
    out_dir: Path,
) -> tuple[Path, Path, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)

    caminho_json = out_dir / "metricas_resumo.json"
    caminho_csv = out_dir / "metricas_ranking_por_k.csv"
    caminho_md = out_dir / "resumo_avaliacao.md"
    caminho_queries = out_dir / "queries_avaliadas.csv"

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
    linhas_md.append(
        f"- **Latência p50 (ms)**: {resumo['metadata']['latencia_inferencia_ms_p50']:.4f}\n"
    )
    linhas_md.append(
        f"- **Latência p95 (ms)**: {resumo['metadata']['latencia_inferencia_ms_p95']:.4f}\n"
    )

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
    parser.add_argument(
        "--model-dir",
        type=str,
        default="treinamento/modelo",
        help="Diretório do modelo/ranker a ser avaliado",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default=str(RESULTADOS_DIR),
        help="Diretório de saída dos arquivos da avaliação",
    )
    args = parser.parse_args()

    ks = _normalizar_k(args.k)
    model_dir = resolve_model_dir(args.model_dir)
    out_dir = Path(args.out_dir)
    if not out_dir.is_absolute():
        out_dir = ROOT / out_dir

    ranker = load_ranker(model_dir)
    resumo, df_metricas, df_queries = avaliar(ranker, ks, model_dir)
    caminho_json, caminho_csv, caminho_md = salvar_resultados(
        resumo, df_metricas, df_queries, out_dir
    )

    print("=== Avaliação concluída ===")
    print(f"Modelo avaliado  : {model_dir}")
    print(f"Consultas avaliadas: {resumo['metadata']['n_queries_validas']}")
    print(f"Resultados JSON : {caminho_json}")
    print(f"Resultados CSV  : {caminho_csv}")
    print(f"Resumo Markdown : {caminho_md}")


if __name__ == "__main__":
    main()
