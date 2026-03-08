"""
Avaliação automática do impacto do sinal de popularidade.

Modo real:
- baseline (peso_popularidade = 0.0)
- depois   (peso_popularidade = valor configurado)
- queries derivadas de `treinamento/dados/splits/val_interactions.parquet`

Modo demo (`--demo`):
- executa benchmark sintético reproduzível para validar o pipeline de avaliação
  mesmo sem artefatos de dados/modelo locais.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from avaliacao.offline_protocol import (
    OfflineQuery,
    build_future_queries,
    load_split_interactions,
    resolve_dataset_dirs,
)
from dataset_context import manifest_path
from pipeline_contracts import split_signature_from_manifest_file, split_signature_from_metadata
from progress_utils import IterationProgress
from treinamento.model_utils import load_model_metadata


def carregar_queries(
    modelo,
    splits_dir: Path,
    output_dir: Path,
    max_queries: int = 200,
    split_name: str = "val",
    seed: int = 42,
) -> list[OfflineQuery]:
    interactions = load_split_interactions(splits_dir, split_name)
    queries = build_future_queries(modelo, interactions, output_dir)
    if max_queries > 0 and len(queries) > max_queries:
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(queries), size=max_queries, replace=False)
        return [queries[int(i)] for i in np.sort(idx)]
    return queries


def precision_at_k(relevantes: set[int], recomendados: list[int], k: int) -> float:
    if k <= 0:
        return 0.0
    rec_k = recomendados[:k]
    if not rec_k:
        return 0.0
    hits = sum(1 for item in rec_k if item in relevantes)
    return hits / k


def ndcg_at_k(relevantes: set[int], recomendados: list[int], k: int) -> float:
    gains = np.array([1.0 if item in relevantes else 0.0 for item in recomendados[:k]], dtype=np.float32)
    if gains.size == 0:
        return 0.0
    discounts = 1.0 / np.log2(np.arange(2, gains.size + 2))
    dcg = float((gains * discounts).sum())
    ideal = np.array([1.0] * min(len(relevantes), k), dtype=np.float32)
    idcg = float((ideal * discounts).sum())
    return dcg / idcg if idcg > 0 else 0.0


def precision_from_rels(rels: list[float], k: int) -> float:
    arr = np.array(rels[:k], dtype=np.float32)
    return float((arr > 0).mean()) if len(arr) else 0.0


def ndcg_from_rels(rels: list[float], k: int) -> float:
    gains = np.array(rels[:k], dtype=np.float32)
    if gains.size == 0:
        return 0.0
    discounts = 1.0 / np.log2(np.arange(2, gains.size + 2))
    dcg = float((gains * discounts).sum())
    ideal = np.sort(gains)[::-1]
    idcg = float((ideal * discounts).sum())
    return dcg / idcg if idcg > 0 else 0.0


def avaliar_real(modelo, queries: list[OfflineQuery], peso_popularidade: float, k: int) -> dict[str, float]:
    precisions, ndcgs = [], []
    progress = IterationProgress(
        total=len(queries),
        label=f"Popularidade real k={k}",
        every_percent=10,
    )
    if queries:
        progress.start("Processando queries")

    for idx, q in enumerate(queries, start=1):
        df = modelo.recommend_df(
            tags=q.reference_tags,
            timestamp=q.reference_timestamp_ms,
            top_k=k,
            excluir_tags_exatas=False,
            peso_popularidade=peso_popularidade,
            include_internal=True,
        )
        rec_ids = []
        if not df.empty and "_message_id" in df.columns:
            rec_ids = (
                df["_message_id"]
                .dropna()
                .astype("int64")
                .tolist()
            )
        precisions.append(precision_at_k(q.future_ids, rec_ids, k))
        ndcgs.append(ndcg_at_k(q.future_ids, rec_ids, k))
        if queries:
            progress.log(idx)

    if queries:
        progress.finish("Avaliação real finalizada")

    return {f"precision@{k}": float(np.mean(precisions)) if precisions else 0.0, f"ndcg@{k}": float(np.mean(ndcgs)) if ndcgs else 0.0, "num_queries": len(queries)}


def avaliar_demo(peso_popularidade: float, k: int, n_queries: int = 200) -> tuple[dict, dict]:
    rng = np.random.default_rng(42)
    n_items = 300
    # Itens: sinais sintéticos em [0,1]
    cosine = rng.uniform(0, 1, size=n_items)
    cooc = rng.uniform(0, 1, size=n_items)
    time_decay = rng.uniform(0, 1, size=n_items)
    social = rng.uniform(0, 1, size=n_items)
    popularity = rng.uniform(0, 1, size=n_items)

    # Relevância latente com contribuição de popularidade (simula cenário em que ela ajuda)
    rel_true = 0.30 * cosine + 0.20 * cooc + 0.15 * time_decay + 0.15 * social + 0.20 * popularity

    def run(weight_pop: float, label: str) -> dict[str, float]:
        precisions, ndcgs = [], []
        progress = IterationProgress(
            total=n_queries,
            label=label,
            every_percent=10,
        )
        if n_queries > 0:
            progress.start("Processando benchmark sintético")

        for idx in range(n_queries):
            noise = rng.normal(0, 0.02, size=n_items)
            score = 0.35 * cosine + 0.25 * cooc + 0.15 * time_decay + 0.15 * social + weight_pop * popularity + noise
            top_idx = np.argsort(score)[::-1][:k]
            rels = rel_true[top_idx].tolist()
            precisions.append(precision_from_rels(rels, k))
            ndcgs.append(ndcg_from_rels(rels, k))
            if n_queries > 0:
                progress.log(idx + 1)

        if n_queries > 0:
            progress.finish("Benchmark sintético finalizado")
        return {f"precision@{k}": float(np.mean(precisions)), f"ndcg@{k}": float(np.mean(ndcgs)), "num_queries": n_queries}

    return (
        run(0.0, f"Popularidade demo antes k={k}"),
        run(peso_popularidade, f"Popularidade demo depois k={k}"),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Avalia impacto do peso de popularidade no ranking.")
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--peso-depois", type=float, default=0.10)
    parser.add_argument("--max-queries", type=int, default=200)
    parser.add_argument("--out-json", type=str, default="avaliacao/metricas_antes_depois.json")
    parser.add_argument("--demo", action="store_true", help="Executa benchmark sintético sem depender de artefatos de treino")
    parser.add_argument(
        "--model-dir",
        type=str,
        default="treinamento/modelo",
        help="Diretório do modelo baseline a ser avaliado",
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

    if args.demo:
        antes, depois = avaliar_demo(args.peso_depois, args.k, args.max_queries)
        modo = "demo"
        metadata_extra = {}
    else:
        from treinamento.recomendar import ModeloRecomendacao
        model_dir = Path(args.model_dir)
        if not model_dir.is_absolute():
            model_dir = (ROOT / model_dir).resolve()
        splits_dir, output_dir = resolve_dataset_dirs(
            model_dir,
            args.dataset_key,
            args.splits_dir,
            args.output_dir,
        )
        modelo = ModeloRecomendacao(model_dir).carregar()
        queries = carregar_queries(
            modelo,
            splits_dir,
            output_dir,
            max_queries=args.max_queries,
            split_name="val",
            seed=42,
        )
        antes = avaliar_real(modelo, queries, 0.0, args.k)
        depois = avaliar_real(modelo, queries, args.peso_depois, args.k)
        modo = "real"
        split_sig = split_signature_from_manifest_file(manifest_path(splits_dir))
        model_split_sig = split_signature_from_metadata(load_model_metadata(model_dir))
        metadata_extra = {
            "split_name": "val",
            "splits_dir": str(splits_dir),
            "output_dir": str(output_dir),
            "split_signature": split_sig,
            "model_split_signature": model_split_sig,
            "split_consistente": bool(split_sig) and bool(model_split_sig) and split_sig == model_split_sig,
        }

    delta = {ch: round(depois[ch] - antes[ch], 6) for ch in antes if ch.startswith("precision") or ch.startswith("ndcg")}
    resultado = {
        "modo": modo,
        "config": {
            "k": args.k,
            "peso_antes": 0.0,
            "peso_depois": args.peso_depois,
        },
        "metadata": metadata_extra,
        "antes": antes,
        "depois": depois,
        "delta": delta,
    }

    out = ROOT / args.out_json
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(resultado, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(resultado, indent=2, ensure_ascii=False))
    print(f"Métricas salvas em: {out}")


if __name__ == "__main__":
    main()
