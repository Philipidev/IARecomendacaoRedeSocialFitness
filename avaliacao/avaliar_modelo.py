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

from avaliacao.offline_protocol import (
    build_future_queries,
    load_split_interactions,
    parse_tags as parse_tags_offline,
    resolve_dataset_dirs,
)
from dataset_context import manifest_path
from pipeline_contracts import split_signature_from_manifest_file, split_signature_from_metadata
from progress_utils import IterationProgress
from treinamento.model_utils import (
    infer_model_family,
    load_model_metadata,
    model_id_from_dir,
    rel_path,
    resolve_model_dir,
)
from treinamento.rankers import load_ranker

RESULTADOS_DIR = ROOT / "avaliacao" / "resultados"
MS_POR_DIA = 86_400_000
K_PADRAO = [5, 10, 20]


def _parse_tags(value) -> list[str]:
    return parse_tags_offline(value)


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
    splits_dir: Path,
    output_dir: Path,
) -> tuple[dict, pd.DataFrame, pd.DataFrame]:
    test_interactions = load_split_interactions(splits_dir, "test")
    queries = build_future_queries(ranker, test_interactions, output_dir)

    catalogo_total = len(ranker.artifacts.posts_cache)
    uso_catalogo = set()
    ilads = []
    novidades = []
    recencias = []
    latencias_ms = []
    total_queries_candidatas = len(queries)
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

    for query in queries:
        processed_queries += 1
        try:
            started = perf_counter()
            rec_ids_message, rec_tags, rec_scores, rec_local_idxs = recomendar_ids(
                ranker,
                tags_referencia=query.reference_tags,
                timestamp_referencia=query.reference_timestamp_ms,
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
                    recencias.append(
                        abs(rec_ts - query.reference_timestamp_ms) / MS_POR_DIA
                    )

            for k in ks:
                acumuladores[k]["precision"].append(
                    precision_at_k(query.future_ids, rec_ids_message, k)
                )
                acumuladores[k]["recall"].append(
                    recall_at_k(query.future_ids, rec_ids_message, k)
                )
                acumuladores[k]["hitrate"].append(
                    hitrate_at_k(query.future_ids, rec_ids_message, k)
                )
                acumuladores[k]["map"].append(map_at_k(query.future_ids, rec_ids_message, k))
                acumuladores[k]["ndcg"].append(
                    ndcg_at_k(query.future_ids, rec_ids_message, k)
                )
                acumuladores[k]["mrr"].append(mrr_at_k(query.future_ids, rec_ids_message, k))

            linhas_q.append(
                {
                    "user_id": int(query.user_id),
                    "message_id_referencia": int(query.reference_message_id),
                    "timestamp_referencia_ms": int(query.reference_timestamp_ms),
                    "n_relevantes_futuros": len(query.future_ids),
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

    current_split_signature = split_signature_from_manifest_file(manifest_path(splits_dir))
    model_metadata = load_model_metadata(model_dir)
    model_split_signature = split_signature_from_metadata(model_metadata)
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
        "split_signature": current_split_signature,
        "model_split_signature": model_split_signature,
        "split_consistente": (
            bool(current_split_signature)
            and bool(model_split_signature)
            and current_split_signature == model_split_signature
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
    parser.add_argument(
        "--dataset-key",
        type=str,
        default=None,
        help="Namespace lógico do dataset; usado como fallback quando o metadata não informa",
    )
    parser.add_argument(
        "--splits-dir",
        type=str,
        default=None,
        help="Override opcional do diretório de splits",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Override opcional do diretório de parquets extraídos",
    )
    args = parser.parse_args()

    ks = _normalizar_k(args.k)
    model_dir = resolve_model_dir(args.model_dir)
    splits_dir, output_dir = resolve_dataset_dirs(
        model_dir,
        args.dataset_key,
        args.splits_dir,
        args.output_dir,
    )
    out_dir = Path(args.out_dir)
    if not out_dir.is_absolute():
        out_dir = ROOT / out_dir

    ranker = load_ranker(model_dir)
    resumo, df_metricas, df_queries = avaliar(
        ranker,
        ks,
        model_dir,
        splits_dir,
        output_dir,
    )
    caminho_json, caminho_csv, caminho_md = salvar_resultados(
        resumo, df_metricas, df_queries, out_dir
    )

    print("=== Avaliação concluída ===")
    print(f"Modelo avaliado  : {model_dir}")
    print(f"Splits utilizados: {splits_dir}")
    print(f"Output utilizado : {output_dir}")
    print(f"Consultas avaliadas: {resumo['metadata']['n_queries_validas']}")
    print(f"Resultados JSON : {caminho_json}")
    print(f"Resultados CSV  : {caminho_csv}")
    print(f"Resumo Markdown : {caminho_md}")


if __name__ == "__main__":
    main()
