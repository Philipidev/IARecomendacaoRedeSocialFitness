"""
Otimização dos pesos do score híbrido da recomendação.

Fluxo:
  1) Carrega dados de validação (treinamento/dados/splits/val_*)
  2) Carrega artefatos do modelo via ModeloRecomendacao
  3) Executa Grid Search com restrição:
       w_cos + w_cooc + w_time + w_social = 1.0 e w_i >= 0
  4) Opcionalmente executa Random Search para refino
  5) Calcula NDCG@K (alvo) e métricas secundárias
  6) Salva ranking em avaliacao/resultados/pesos_experimentos.csv
  7) Exporta melhor configuração em treinamento/modelo/pesos_otimos.json

Métrica alvo (NDCG@K):
  Para cada post de validação usado como consulta, os candidatos recomendados
  recebem ganho pela similaridade Jaccard entre tags da consulta e tags do post
  recomendado. O DCG da lista prevista é comparado ao DCG ideal (ordenação por
  ganho real), gerando NDCG em [0, 1].
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from treinamento.recomendar import MODELO_DIR, ModeloRecomendacao

SPLITS_DIR = ROOT / "treinamento" / "dados" / "splits"
RESULTADOS_PATH = ROOT / "avaliacao" / "resultados" / "pesos_experimentos.csv"
PESOS_OTIMOS_PATH = MODELO_DIR / "pesos_otimos.json"


@dataclass
class ResultadoPesos:
    w_cos: float
    w_cooc: float
    w_time: float
    w_social: float
    ndcg_at_k: float
    precision_at_k: float
    avg_jaccard_at_k: float
    cobertura_top_k: float
    avaliadas: int


def _parse_tags(value) -> list[str]:
    if isinstance(value, list):
        return [str(v) for v in value]
    if isinstance(value, np.ndarray):
        return [str(v) for v in value.tolist()]
    if isinstance(value, str):
        txt = value.strip()
        if txt.startswith("[") and txt.endswith("]"):
            try:
                parsed = json.loads(txt.replace("'", '"'))
                if isinstance(parsed, list):
                    return [str(v) for v in parsed]
            except Exception:
                pass
        return [txt]
    return []


def carregar_validacao() -> pd.DataFrame:
    val_files = sorted(SPLITS_DIR.glob("val_*.parquet"))
    if not val_files:
        raise FileNotFoundError(
            f"Nenhum arquivo val_*.parquet encontrado em {SPLITS_DIR}. "
            "Execute primeiro: python treinamento/dividir_dataset.py"
        )

    val_posts = SPLITS_DIR / "val_posts.parquet"
    if not val_posts.exists():
        raise FileNotFoundError(
            f"Arquivo obrigatório ausente: {val_posts}. "
            "Gere os splits com python treinamento/dividir_dataset.py"
        )

    df_val = pd.read_parquet(val_posts)
    if "tags_fitness" not in df_val.columns or "creation_date" not in df_val.columns:
        raise ValueError("val_posts.parquet deve conter colunas tags_fitness e creation_date.")

    df_val = df_val.copy()
    df_val["tags_fitness"] = df_val["tags_fitness"].apply(_parse_tags)
    df_val = df_val[df_val["tags_fitness"].apply(bool)]
    return df_val.reset_index(drop=True)


def gerar_combinacoes_grid(passo: float) -> list[tuple[float, float, float, float]]:
    if passo <= 0 or passo > 1:
        raise ValueError("--grid-step deve estar no intervalo (0, 1].")

    n = int(round(1.0 / passo))
    if not math.isclose(n * passo, 1.0, rel_tol=0, abs_tol=1e-8):
        raise ValueError("--grid-step deve dividir 1.0 exatamente (ex.: 0.1, 0.05, 0.02).")

    combinacoes: list[tuple[float, float, float, float]] = []
    for a in range(n + 1):
        w_cos = a * passo
        for b in range(n + 1 - a):
            w_cooc = b * passo
            for c in range(n + 1 - a - b):
                w_time = c * passo
                w_social = 1.0 - w_cos - w_cooc - w_time
                if w_social < -1e-9:
                    continue
                combinacoes.append(
                    (
                        round(max(w_cos, 0.0), 8),
                        round(max(w_cooc, 0.0), 8),
                        round(max(w_time, 0.0), 8),
                        round(max(w_social, 0.0), 8),
                    )
                )
    return combinacoes


def random_vizinhos_dirichlet(
    base: tuple[float, float, float, float],
    n_amostras: int,
    rng: np.random.Generator,
    concentracao: float,
) -> list[tuple[float, float, float, float]]:
    if n_amostras <= 0:
        return []

    alpha_base = np.array(base, dtype=np.float64) * concentracao + 1e-3
    draws = rng.dirichlet(alpha_base, size=n_amostras)
    return [tuple(float(round(v, 8)) for v in row) for row in draws]


def _jaccard(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    inter = len(a & b)
    uni = len(a | b)
    return inter / uni if uni > 0 else 0.0


def _dcg(rels: np.ndarray) -> float:
    if rels.size == 0:
        return 0.0
    discounts = 1.0 / np.log2(np.arange(2, rels.size + 2))
    return float(np.sum(rels * discounts))


def avaliar_pesos(
    modelo: ModeloRecomendacao,
    validacao: pd.DataFrame,
    pesos: tuple[float, float, float, float],
    top_k: int,
    max_queries: int,
    rng: np.random.Generator,
) -> ResultadoPesos:
    w_cos, w_cooc, w_time, w_social = pesos
    if any(w < 0 for w in pesos):
        raise ValueError("Pesos inválidos: todos devem ser >= 0.")
    if not math.isclose(sum(pesos), 1.0, rel_tol=0, abs_tol=1e-6):
        raise ValueError("Pesos inválidos: soma deve ser 1.0.")

    if max_queries > 0 and len(validacao) > max_queries:
        idx = rng.choice(len(validacao), size=max_queries, replace=False)
        consultas = validacao.iloc[np.sort(idx)]
    else:
        consultas = validacao

    ndcgs: list[float] = []
    precisions: list[float] = []
    avg_jaccards: list[float] = []
    all_recomendados: set[int] = set()

    for _, row in consultas.iterrows():
        tags_q = [t.strip() for t in row["tags_fitness"] if str(t).strip()]
        if not tags_q:
            continue

        sc = modelo._score_cosine(tags_q)
        si = modelo._score_cooccurrence(tags_q)
        st = modelo._score_time_decay(int(row["creation_date"]))
        ss = modelo._score_social()

        score = w_cos * sc + w_cooc * si + w_time * st + w_social * ss
        ordem = np.argsort(-score)
        top_idx = ordem[:top_k]
        all_recomendados.update(top_idx.tolist())

        tags_query_set = set(tags_q)
        gains = np.array(
            [_jaccard(tags_query_set, set(modelo._posts.iloc[i]["tags_fitness"])) for i in top_idx],
            dtype=np.float64,
        )
        ideal = np.sort(gains)[::-1]

        dcg = _dcg(gains)
        idcg = _dcg(ideal)
        ndcg = (dcg / idcg) if idcg > 0 else 0.0

        prec = float(np.mean(gains > 0)) if gains.size else 0.0
        avg_j = float(np.mean(gains)) if gains.size else 0.0

        ndcgs.append(ndcg)
        precisions.append(prec)
        avg_jaccards.append(avg_j)

    cobertura = len(all_recomendados) / len(modelo._posts) if len(modelo._posts) else 0.0

    return ResultadoPesos(
        w_cos=w_cos,
        w_cooc=w_cooc,
        w_time=w_time,
        w_social=w_social,
        ndcg_at_k=float(np.mean(ndcgs)) if ndcgs else 0.0,
        precision_at_k=float(np.mean(precisions)) if precisions else 0.0,
        avg_jaccard_at_k=float(np.mean(avg_jaccards)) if avg_jaccards else 0.0,
        cobertura_top_k=float(cobertura),
        avaliadas=len(ndcgs),
    )


def salvar_resultados(df: pd.DataFrame) -> None:
    RESULTADOS_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(RESULTADOS_PATH, index=False)


def salvar_melhor_peso(linha: pd.Series, top_k: int, grid_step: float, random_search: int) -> None:
    MODELO_DIR.mkdir(parents=True, exist_ok=True)
    payload = {
        "w_cos": float(linha["w_cos"]),
        "w_cooc": float(linha["w_cooc"]),
        "w_time": float(linha["w_time"]),
        "w_social": float(linha["w_social"]),
        "metric_target": "ndcg_at_k",
        "metric_target_value": float(linha["ndcg_at_k"]),
        "top_k": int(top_k),
        "otimizacao": {
            "grid_step": grid_step,
            "random_search_amostras": random_search,
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        },
    }
    with open(PESOS_OTIMOS_PATH, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def main() -> None:
    parser = argparse.ArgumentParser(description="Otimiza pesos do score híbrido por validação.")
    parser.add_argument("--grid-step", type=float, default=0.1, help="Passo do grid search (ex.: 0.1 ou 0.05)")
    parser.add_argument("--random-search", type=int, default=0, help="Número de amostras aleatórias para refino")
    parser.add_argument("--random-alpha", type=float, default=80.0, help="Concentração Dirichlet no refino")
    parser.add_argument("--top-k", type=int, default=10, help="Top-K da métrica NDCG@K")
    parser.add_argument("--max-queries", type=int, default=300, help="Limite de consultas da validação (0 = todas)")
    parser.add_argument("--seed", type=int, default=42, help="Seed reprodutível")
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)

    print("Carregando dados de validação...")
    validacao = carregar_validacao()
    print(f"  Consultas válidas: {len(validacao)}")

    print("Carregando artefatos do modelo...")
    modelo = ModeloRecomendacao().carregar()

    print("Executando Grid Search...")
    combinacoes = gerar_combinacoes_grid(args.grid_step)
    print(f"  Combinações de grid: {len(combinacoes)}")

    resultados: list[ResultadoPesos] = []
    for pesos in combinacoes:
        resultados.append(
            avaliar_pesos(
                modelo=modelo,
                validacao=validacao,
                pesos=pesos,
                top_k=args.top_k,
                max_queries=args.max_queries,
                rng=rng,
            )
        )

    if args.random_search > 0 and resultados:
        print("Executando Random Search (refino)...")
        top_base = sorted(resultados, key=lambda r: r.ndcg_at_k, reverse=True)[:5]
        amostras_por_base = max(1, args.random_search // len(top_base))
        extras: list[tuple[float, float, float, float]] = []
        for base in top_base:
            extras.extend(
                random_vizinhos_dirichlet(
                    base=(base.w_cos, base.w_cooc, base.w_time, base.w_social),
                    n_amostras=amostras_por_base,
                    rng=rng,
                    concentracao=args.random_alpha,
                )
            )

        vistos = {(r.w_cos, r.w_cooc, r.w_time, r.w_social) for r in resultados}
        for pesos in extras:
            soma = sum(pesos)
            if not math.isclose(soma, 1.0, rel_tol=0, abs_tol=1e-4):
                continue
            if pesos in vistos:
                continue
            vistos.add(pesos)
            resultados.append(
                avaliar_pesos(
                    modelo=modelo,
                    validacao=validacao,
                    pesos=pesos,
                    top_k=args.top_k,
                    max_queries=args.max_queries,
                    rng=rng,
                )
            )

    df_res = pd.DataFrame([r.__dict__ for r in resultados])
    if df_res.empty:
        raise RuntimeError("Nenhuma combinação foi avaliada.")

    df_res["sum_w"] = df_res[["w_cos", "w_cooc", "w_time", "w_social"]].sum(axis=1)
    df_res = df_res.sort_values(
        by=["ndcg_at_k", "precision_at_k", "avg_jaccard_at_k", "cobertura_top_k"],
        ascending=False,
    ).reset_index(drop=True)

    salvar_resultados(df_res)
    melhor = df_res.iloc[0]
    salvar_melhor_peso(melhor, top_k=args.top_k, grid_step=args.grid_step, random_search=args.random_search)

    print("\n=== Melhor configuração ===")
    print(
        f"w_cos={melhor['w_cos']:.4f}, w_cooc={melhor['w_cooc']:.4f}, "
        f"w_time={melhor['w_time']:.4f}, w_social={melhor['w_social']:.4f}"
    )
    print(f"NDCG@{args.top_k}: {melhor['ndcg_at_k']:.4f}")
    print(f"Precision@{args.top_k}: {melhor['precision_at_k']:.4f}")
    print(f"Avg Jaccard@{args.top_k}: {melhor['avg_jaccard_at_k']:.4f}")
    print(f"Resultados salvos em: {RESULTADOS_PATH}")
    print(f"Pesos ótimos exportados em: {PESOS_OTIMOS_PATH}")


if __name__ == "__main__":
    main()
