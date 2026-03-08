"""
Otimização dos pesos do score híbrido da recomendação.

Fluxo:
  1) Carrega queries do split de validação (protocolo offline por interações futuras)
  2) Carrega artefatos do modelo via ModeloRecomendacao
  3) Executa Grid Search com restrição:
       w_cos + w_cooc + w_time + w_social = 1.0 e w_i >= 0
  4) Opcionalmente executa Random Search para refino
  5) Calcula NDCG@K (alvo) e métricas secundárias
  6) Salva ranking em avaliacao/resultados/pesos_experimentos.csv
  7) Exporta melhor configuração em treinamento/modelo/pesos_otimos.json

Métrica alvo (NDCG@K):
  Usa o mesmo protocolo da avaliação offline principal: cada interação de validação
  vira item de referência, e o ground truth é o conjunto de interações futuras reais
  do mesmo usuário ainda presentes no catálogo do modelo.
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

from avaliacao.offline_protocol import (
    OfflineQuery,
    build_future_queries,
    load_split_interactions,
    resolve_dataset_dirs,
)
from dataset_context import manifest_path
from pipeline_contracts import split_signature_from_manifest_file, split_signature_from_metadata
from progress_utils import IterationProgress
from treinamento.model_utils import load_model_metadata, resolve_model_dir
from treinamento.recomendar import MODELO_DIR, ModeloRecomendacao

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
    recall_at_k: float
    cobertura_top_k: float
    avaliadas: int


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


def carregar_validacao(
    modelo,
    splits_dir: Path,
    output_dir: Path,
    max_queries: int,
    seed: int,
) -> list[OfflineQuery]:
    interactions = load_split_interactions(splits_dir, "val")
    queries = build_future_queries(modelo, interactions, output_dir)
    if max_queries > 0 and len(queries) > max_queries:
        idx = np.random.default_rng(seed).choice(
            len(queries),
            size=max_queries,
            replace=False,
        )
        return [queries[int(i)] for i in np.sort(idx)]
    return queries


def _precision_at_k(relevantes: set[int], recomendados: list[int], k: int) -> float:
    if k <= 0:
        return 0.0
    rec_k = recomendados[:k]
    if not rec_k:
        return 0.0
    hits = sum(1 for item in rec_k if item in relevantes)
    return hits / k


def _recall_at_k(relevantes: set[int], recomendados: list[int], k: int) -> float:
    if not relevantes:
        return 0.0
    rec_k = recomendados[:k]
    hits = sum(1 for item in rec_k if item in relevantes)
    return hits / len(relevantes)


def _ndcg_at_k(relevantes: set[int], recomendados: list[int], k: int) -> float:
    gains = np.array(
        [1.0 if item in relevantes else 0.0 for item in recomendados[:k]],
        dtype=np.float64,
    )
    if gains.size == 0:
        return 0.0
    discounts = 1.0 / np.log2(np.arange(2, gains.size + 2))
    dcg = float(np.sum(gains * discounts))
    ideal = np.array([1.0] * min(len(relevantes), k), dtype=np.float64)
    idcg = float(np.sum(ideal * discounts[: len(ideal)]))
    return (dcg / idcg) if idcg > 0 else 0.0


def avaliar_pesos(
    modelo: ModeloRecomendacao,
    validacao: list[OfflineQuery],
    pesos: tuple[float, float, float, float],
    top_k: int,
) -> ResultadoPesos:
    w_cos, w_cooc, w_time, w_social = pesos
    if any(w < 0 for w in pesos):
        raise ValueError("Pesos inválidos: todos devem ser >= 0.")
    if not math.isclose(sum(pesos), 1.0, rel_tol=0, abs_tol=1e-6):
        raise ValueError("Pesos inválidos: soma deve ser 1.0.")

    ndcgs: list[float] = []
    precisions: list[float] = []
    recalls: list[float] = []
    all_recomendados: set[int] = set()
    catalog_message_ids = (
        pd.to_numeric(modelo._posts["_message_id"], errors="coerce")
        .fillna(-1)
        .astype("int64")
        .to_numpy()
    )

    for query in validacao:
        sc = modelo._score_cosine(query.reference_tags)
        si = modelo._score_cooccurrence(query.reference_tags)
        st = modelo._score_time_decay(query.reference_timestamp_ms)
        ss = modelo._score_social()

        score = w_cos * sc + w_cooc * si + w_time * st + w_social * ss
        ordem = np.argsort(-score)
        top_idx = ordem[:top_k]
        all_recomendados.update(top_idx.tolist())
        rec_ids = [
            int(catalog_message_ids[i])
            for i in top_idx
            if i < len(catalog_message_ids) and int(catalog_message_ids[i]) >= 0
        ]

        ndcgs.append(_ndcg_at_k(query.future_ids, rec_ids, top_k))
        precisions.append(_precision_at_k(query.future_ids, rec_ids, top_k))
        recalls.append(_recall_at_k(query.future_ids, rec_ids, top_k))

    cobertura = len(all_recomendados) / len(modelo._posts) if len(modelo._posts) else 0.0

    return ResultadoPesos(
        w_cos=w_cos,
        w_cooc=w_cooc,
        w_time=w_time,
        w_social=w_social,
        ndcg_at_k=float(np.mean(ndcgs)) if ndcgs else 0.0,
        precision_at_k=float(np.mean(precisions)) if precisions else 0.0,
        recall_at_k=float(np.mean(recalls)) if recalls else 0.0,
        cobertura_top_k=float(cobertura),
        avaliadas=len(ndcgs),
    )


def salvar_resultados(df: pd.DataFrame, out_csv: Path) -> None:
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)


def salvar_melhor_peso(
    linha: pd.Series,
    top_k: int,
    grid_step: float,
    random_search: int,
    split_signature: str | None,
    model_split_signature: str | None,
    pesos_path: Path,
) -> None:
    pesos_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "w_cos": float(linha["w_cos"]),
        "w_cooc": float(linha["w_cooc"]),
        "w_time": float(linha["w_time"]),
        "w_social": float(linha["w_social"]),
        "metric_target": "ndcg_at_k",
        "metric_target_value": float(linha["ndcg_at_k"]),
        "top_k": int(top_k),
        "otimizacao": {
            "protocol": "offline_future_interactions_val",
            "grid_step": grid_step,
            "random_search_amostras": random_search,
            "split_signature": split_signature,
            "model_split_signature": model_split_signature,
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        },
    }
    with open(pesos_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def main() -> None:
    parser = argparse.ArgumentParser(description="Otimiza pesos do score híbrido por validação.")
    parser.add_argument("--grid-step", type=float, default=0.1, help="Passo do grid search (ex.: 0.1 ou 0.05)")
    parser.add_argument("--random-search", type=int, default=0, help="Número de amostras aleatórias para refino")
    parser.add_argument("--random-alpha", type=float, default=80.0, help="Concentração Dirichlet no refino")
    parser.add_argument("--top-k", type=int, default=10, help="Top-K da métrica NDCG@K")
    parser.add_argument("--max-queries", type=int, default=300, help="Limite de consultas da validação (0 = todas)")
    parser.add_argument("--seed", type=int, default=42, help="Seed reprodutível")
    parser.add_argument(
        "--model-dir",
        type=str,
        default=str(MODELO_DIR),
        help="Diretório do modelo baseline a ser otimizado",
    )
    parser.add_argument(
        "--out-csv",
        type=str,
        default=str(RESULTADOS_PATH),
        help="Arquivo CSV de saída com os experimentos avaliados",
    )
    parser.add_argument(
        "--out-json",
        type=str,
        default=None,
        help="Arquivo JSON opcional para salvar os pesos ótimos",
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

    rng = np.random.default_rng(args.seed)
    model_dir = resolve_model_dir(args.model_dir)
    splits_dir, output_dir = resolve_dataset_dirs(
        model_dir,
        args.dataset_key,
        args.splits_dir,
        args.output_dir,
    )
    out_csv = Path(args.out_csv)
    if not out_csv.is_absolute():
        out_csv = ROOT / out_csv
    pesos_path = Path(args.out_json) if args.out_json else (model_dir / "pesos_otimos.json")
    if not pesos_path.is_absolute():
        pesos_path = ROOT / pesos_path

    print("Carregando artefatos do modelo...")
    modelo = ModeloRecomendacao(model_dir).carregar()

    print("Carregando dados de validação...")
    validacao = carregar_validacao(
        modelo,
        splits_dir,
        output_dir,
        args.max_queries,
        args.seed,
    )
    print(f"  Consultas válidas: {len(validacao)}")
    print(f"  Splits utilizados: {splits_dir}")
    print(f"  Output utilizado : {output_dir}")

    print("Executando Grid Search...")
    combinacoes = gerar_combinacoes_grid(args.grid_step)
    print(f"  Combinações de grid: {len(combinacoes)}")

    resultados: list[ResultadoPesos] = []
    grid_progress = IterationProgress(
        total=len(combinacoes),
        label="Otimização grid",
        every_percent=5,
    )
    if combinacoes:
        grid_progress.start("Avaliando combinações do grid")

    for idx, pesos in enumerate(combinacoes, start=1):
        resultados.append(
            avaliar_pesos(
                modelo=modelo,
                validacao=validacao,
                pesos=pesos,
                top_k=args.top_k,
            )
        )
        if combinacoes:
            grid_progress.log(idx)

    if combinacoes:
        grid_progress.finish("Grid Search finalizado")

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
        random_progress = IterationProgress(
            total=len(extras),
            label="Otimização random",
            every_percent=10,
        )
        if extras:
            random_progress.start("Avaliando amostras do refino")

        for idx, pesos in enumerate(extras, start=1):
            soma = sum(pesos)
            if not math.isclose(soma, 1.0, rel_tol=0, abs_tol=1e-4):
                random_progress.log(idx)
                continue
            if pesos in vistos:
                random_progress.log(idx)
                continue
            vistos.add(pesos)
            resultados.append(
                avaliar_pesos(
                    modelo=modelo,
                    validacao=validacao,
                    pesos=pesos,
                    top_k=args.top_k,
                )
            )
            random_progress.log(idx)

        if extras:
            random_progress.finish("Random Search finalizado")

    df_res = pd.DataFrame([r.__dict__ for r in resultados])
    if df_res.empty:
        raise RuntimeError("Nenhuma combinação foi avaliada.")

    df_res["sum_w"] = df_res[["w_cos", "w_cooc", "w_time", "w_social"]].sum(axis=1)
    df_res = df_res.sort_values(
        by=["ndcg_at_k", "precision_at_k", "recall_at_k", "cobertura_top_k"],
        ascending=False,
    ).reset_index(drop=True)

    salvar_resultados(df_res, out_csv)
    melhor = df_res.iloc[0]
    salvar_melhor_peso(
        melhor,
        top_k=args.top_k,
        grid_step=args.grid_step,
        random_search=args.random_search,
        split_signature=split_signature_from_manifest_file(manifest_path(splits_dir)),
        model_split_signature=split_signature_from_metadata(load_model_metadata(model_dir)),
        pesos_path=pesos_path,
    )

    print("\n=== Melhor configuração ===")
    print(
        f"w_cos={melhor['w_cos']:.4f}, w_cooc={melhor['w_cooc']:.4f}, "
        f"w_time={melhor['w_time']:.4f}, w_social={melhor['w_social']:.4f}"
    )
    print(f"NDCG@{args.top_k}: {melhor['ndcg_at_k']:.4f}")
    print(f"Precision@{args.top_k}: {melhor['precision_at_k']:.4f}")
    print(f"Recall@{args.top_k}: {melhor['recall_at_k']:.4f}")
    print(f"Resultados salvos em: {out_csv}")
    print(f"Pesos ótimos exportados em: {pesos_path}")


if __name__ == "__main__":
    main()
