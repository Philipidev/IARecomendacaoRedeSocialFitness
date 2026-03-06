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
import ast
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent.parent
SPLITS_DIR = ROOT / "treinamento" / "dados" / "splits"


@dataclass
class Query:
    tags: list[str]
    timestamp: int


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


def carregar_queries(max_queries: int = 200) -> list[Query]:
    caminho = SPLITS_DIR / "val_interactions.parquet"
    if not caminho.exists():
        raise FileNotFoundError(f"Arquivo ausente: {caminho}")

    df = pd.read_parquet(caminho)
    df = df.copy()
    df["tags_fitness"] = df["tags_fitness"].apply(_parse_tags)
    df = df[df["tags_fitness"].map(len) > 0]
    if len(df) == 0:
        return []

    df = df.sample(n=min(max_queries, len(df)), random_state=42)
    return [Query(tags=row.tags_fitness, timestamp=int(row.timestamp)) for row in df.itertuples()]


def precision_at_k(rels: list[float], k: int) -> float:
    arr = np.array(rels[:k], dtype=np.float32)
    return float((arr > 0).mean()) if len(arr) else 0.0


def ndcg_at_k(rels: list[float], k: int) -> float:
    gains = np.array(rels[:k], dtype=np.float32)
    if gains.size == 0:
        return 0.0
    discounts = 1.0 / np.log2(np.arange(2, gains.size + 2))
    dcg = float((gains * discounts).sum())
    ideal = np.sort(gains)[::-1]
    idcg = float((ideal * discounts).sum())
    return dcg / idcg if idcg > 0 else 0.0


def avaliar_real(modelo, queries: list[Query], peso_popularidade: float, k: int) -> dict[str, float]:
    precisions, ndcgs = [], []
    for q in queries:
        df = modelo.recomendar(tags=q.tags, timestamp=q.timestamp, top_k=k, excluir_tags_exatas=False, peso_popularidade=peso_popularidade)
        query_tags = set(q.tags)
        rels = []
        for tags_post in df.get("tags_fitness", []).tolist() if not df.empty else []:
            tags_post_set = set(tags_post if isinstance(tags_post, list) else _parse_tags(tags_post))
            inter = len(query_tags.intersection(tags_post_set))
            uni = max(len(query_tags.union(tags_post_set)), 1)
            rels.append(inter / uni)
        precisions.append(precision_at_k(rels, k))
        ndcgs.append(ndcg_at_k(rels, k))

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

    def run(weight_pop: float) -> dict[str, float]:
        precisions, ndcgs = [], []
        for _ in range(n_queries):
            noise = rng.normal(0, 0.02, size=n_items)
            score = 0.35 * cosine + 0.25 * cooc + 0.15 * time_decay + 0.15 * social + weight_pop * popularity + noise
            top_idx = np.argsort(score)[::-1][:k]
            rels = rel_true[top_idx].tolist()
            precisions.append(precision_at_k(rels, k))
            ndcgs.append(ndcg_at_k(rels, k))
        return {f"precision@{k}": float(np.mean(precisions)), f"ndcg@{k}": float(np.mean(ndcgs)), "num_queries": n_queries}

    return run(0.0), run(peso_popularidade)


def main() -> None:
    parser = argparse.ArgumentParser(description="Avalia impacto do peso de popularidade no ranking.")
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--peso-depois", type=float, default=0.10)
    parser.add_argument("--max-queries", type=int, default=200)
    parser.add_argument("--out-json", type=str, default="avaliacao/metricas_antes_depois.json")
    parser.add_argument("--demo", action="store_true", help="Executa benchmark sintético sem depender de artefatos de treino")
    args = parser.parse_args()

    if args.demo:
        antes, depois = avaliar_demo(args.peso_depois, args.k, args.max_queries)
        modo = "demo"
    else:
        import sys
        sys.path.insert(0, str(ROOT))
        from treinamento.recomendar import ModeloRecomendacao
        modelo = ModeloRecomendacao().carregar()
        queries = carregar_queries(max_queries=args.max_queries)
        antes = avaliar_real(modelo, queries, 0.0, args.k)
        depois = avaliar_real(modelo, queries, args.peso_depois, args.k)
        modo = "real"

    delta = {ch: round(depois[ch] - antes[ch], 6) for ch in antes if ch.startswith("precision") or ch.startswith("ndcg")}
    resultado = {"modo": modo, "config": {"k": args.k, "peso_antes": 0.0, "peso_depois": args.peso_depois}, "antes": antes, "depois": depois, "delta": delta}

    out = ROOT / args.out_json
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(resultado, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(resultado, indent=2, ensure_ascii=False))
    print(f"Métricas salvas em: {out}")


if __name__ == "__main__":
    main()
