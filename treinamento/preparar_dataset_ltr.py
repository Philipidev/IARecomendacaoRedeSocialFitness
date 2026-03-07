from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from progress_utils import IterationProgress, StageProgress
from treinamento.model_utils import rel_path, resolve_model_dir, write_json
from treinamento.ranker_features import (
    BaseArtifacts,
    build_categorical_maps,
    build_feature_frame,
    load_base_artifacts,
    parse_tags,
)

SPLITS_DIR = ROOT / "treinamento" / "dados" / "splits"

DEFAULT_FEATURES = [
    "cosine_score",
    "cooccurrence_score",
    "time_decay_score",
    "social_score",
    "popularidade_score",
    "tag_overlap_count",
    "tag_jaccard",
    "num_tags_candidate",
    "content_length",
    "message_type_code",
    "language_code",
]


def _resolve_output_path(path_str: str | None, default_path: Path) -> Path:
    if not path_str:
        return default_path
    path = Path(path_str)
    return path if path.is_absolute() else ROOT / path


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


def _timestamp_ms(value: Any) -> int | None:
    if pd.isna(value):
        return None
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        return int(value)
    try:
        dt = pd.to_datetime(str(value), utc=True)
        return int(dt.value // 1_000_000)
    except Exception:
        return None


def _load_split_interactions(split_name: str) -> pd.DataFrame:
    path = SPLITS_DIR / f"{split_name}_interactions.parquet"
    if not path.exists():
        raise FileNotFoundError(
            f"Arquivo ausente: {path}. Execute primeiro python treinamento/dividir_dataset.py"
        )
    df = pd.read_parquet(path)
    tempo_col = _detectar_coluna_tempo(df)
    if tempo_col is None:
        raise ValueError(f"{path.name} não possui coluna temporal reconhecida.")
    df = df.copy()
    df["__ts_ms"] = df[tempo_col].apply(_timestamp_ms)
    if df["__ts_ms"].isna().all():
        df["__ts_ms"] = np.arange(len(df), dtype=np.int64)
    df["message_id"] = pd.to_numeric(df["message_id"], errors="coerce").astype("Int64")
    df = df.dropna(subset=["message_id"]).copy()
    df["message_id"] = df["message_id"].astype("int64")
    return df


def _build_message_to_rowpos(posts_cache: pd.DataFrame) -> dict[int, int]:
    if "_message_id" not in posts_cache.columns:
        raise ValueError(
            "posts_cache.parquet não contém _message_id. "
            "Regenere os splits e treine com catálogo rastreável."
        )

    message_to_rowpos: dict[int, int] = {}
    values = pd.to_numeric(posts_cache["_message_id"], errors="coerce")
    for row_pos, value in enumerate(values):
        if pd.notna(value):
            message_to_rowpos[int(value)] = row_pos
    return message_to_rowpos


def _sample_queries(
    split_name: str,
    interactions: pd.DataFrame,
    message_to_rowpos: dict[int, int],
    posts_cache: pd.DataFrame,
    max_queries: int,
    rng: np.random.Generator,
) -> list[dict[str, Any]]:
    queries: list[dict[str, Any]] = []
    total_users = int(interactions["user_id"].nunique()) if "user_id" in interactions.columns else 0
    progress = IterationProgress(
        total=total_users,
        label=f"Queries {split_name}",
        every_percent=10,
    )

    if total_users > 0:
        progress.start("Varrendo usuários para gerar queries")

    for processed_users, (user_id, group) in enumerate(interactions.groupby("user_id"), start=1):
        ordered = group.sort_values("__ts_ms")
        eventos = ordered[["message_id", "__ts_ms"]].to_dict("records")
        if len(eventos) < 2:
            progress.log(processed_users)
            continue

        for i, evento in enumerate(eventos[:-1]):
            ref_message_id = int(evento["message_id"])
            if ref_message_id not in message_to_rowpos:
                continue

            future_ids = {
                int(item["message_id"])
                for item in eventos[i + 1 :]
                if int(item["message_id"]) in message_to_rowpos
            }
            if not future_ids:
                continue

            ref_row = posts_cache.iloc[message_to_rowpos[ref_message_id]]
            tags_ref = parse_tags(ref_row.get("tags_fitness", []))
            if not tags_ref:
                continue

            timestamp_ref = _timestamp_ms(ref_row.get("creation_date"))
            if timestamp_ref is None:
                timestamp_ref = int(evento["__ts_ms"])

            queries.append(
                {
                    "user_id": int(user_id),
                    "query_message_id": ref_message_id,
                    "future_ids": future_ids,
                    "tags_ref": tags_ref,
                    "timestamp_ref": int(timestamp_ref),
                }
            )

        progress.log(processed_users)

    if total_users > 0:
        progress.finish(f"Queries candidatas geradas: {len(queries)}")

    if max_queries > 0 and len(queries) > max_queries:
        sampled_idx = rng.choice(len(queries), size=max_queries, replace=False)
        return [queries[int(idx)] for idx in sorted(sampled_idx)]
    return queries


def _build_query_rows(
    split_name: str,
    artifacts: BaseArtifacts,
    interactions: pd.DataFrame,
    features_enabled: list[str],
    negatives_per_query: int,
    hard_negative_topn: int,
    max_queries: int,
    rng: np.random.Generator,
) -> pd.DataFrame:
    posts_cache = artifacts.posts_cache
    message_to_rowpos = _build_message_to_rowpos(posts_cache)
    categorical_maps = build_categorical_maps(posts_cache)
    queries = _sample_queries(
        split_name=split_name,
        interactions=interactions,
        message_to_rowpos=message_to_rowpos,
        posts_cache=posts_cache,
        max_queries=max_queries,
        rng=rng,
    )

    if not queries:
        print(f"[Linhas LTR {split_name}] 0/0 (100 %) - Nenhuma query disponível")
        return pd.DataFrame()

    rows: list[dict[str, Any]] = []
    progress = IterationProgress(
        total=len(queries),
        label=f"Linhas LTR {split_name}",
        every_percent=5,
    )
    progress.start("Montando pares query-item")

    for query_id, query in enumerate(queries):
        features_df = build_feature_frame(
            artifacts,
            tags_entrada=query["tags_ref"],
            timestamp_entrada=query["timestamp_ref"],
            categorical_maps=categorical_maps,
        )

        features_df = features_df[features_df["candidate_message_id"] >= 0].copy()
        features_df = features_df[
            features_df["candidate_message_id"] != query["query_message_id"]
        ].copy()
        if features_df.empty:
            continue

        positive_mask = features_df["candidate_message_id"].isin(query["future_ids"])
        positives = features_df[positive_mask].copy()
        if positives.empty:
            continue

        negatives = features_df[~positive_mask].copy()
        negatives = negatives.sort_values("baseline_score", ascending=False)
        if hard_negative_topn > 0:
            negatives = negatives.head(hard_negative_topn)
        if negatives_per_query > 0 and len(negatives) > negatives_per_query:
            sampled = rng.choice(len(negatives), size=negatives_per_query, replace=False)
            negatives = negatives.iloc[np.sort(sampled)].copy()

        for label, frame in [(1, positives), (0, negatives)]:
            for item in frame.itertuples(index=False):
                row = {
                    "split": split_name,
                    "query_id": int(query_id),
                    "query_user_id": int(query["user_id"]),
                    "query_message_id": int(query["query_message_id"]),
                    "query_timestamp_ms": int(query["timestamp_ref"]),
                    "candidate_message_id": int(item.candidate_message_id),
                    "candidate_catalog_index": int(item.catalog_index),
                    "label": int(label),
                }
                for feature in features_enabled:
                    row[feature] = float(getattr(item, feature))
                rows.append(row)

        progress.log(query_id + 1)

    progress.finish(f"Linhas geradas: {len(rows)}")
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepara datasets query-item para treinamento do ranker LTR."
    )
    parser.add_argument(
        "--model-dir",
        type=str,
        default="treinamento/modelo",
        help="Diretório do modelo base com artefatos do catálogo",
    )
    parser.add_argument(
        "--train-out",
        type=str,
        default=None,
        help="Arquivo parquet de saída do dataset de treino",
    )
    parser.add_argument(
        "--val-out",
        type=str,
        default=None,
        help="Arquivo parquet de saída do dataset de validação",
    )
    parser.add_argument(
        "--meta-out",
        type=str,
        default=None,
        help="Arquivo JSON com schema e estatísticas do dataset LTR",
    )
    parser.add_argument(
        "--features",
        nargs="+",
        default=DEFAULT_FEATURES,
        help="Lista de features usadas no LTR",
    )
    parser.add_argument(
        "--negatives-per-query",
        type=int,
        default=50,
        help="Quantidade de negativos amostrados por query (0 = todos)",
    )
    parser.add_argument(
        "--hard-negative-topn",
        type=int,
        default=500,
        help="Pool dos negativos mais difíceis considerado antes da amostragem",
    )
    parser.add_argument(
        "--max-queries-train",
        type=int,
        default=500,
        help="Limite de queries do split de treino (0 = todas)",
    )
    parser.add_argument(
        "--max-queries-val",
        type=int,
        default=200,
        help="Limite de queries do split de validação (0 = todas)",
    )
    parser.add_argument("--seed", type=int, default=42, help="Seed reprodutível")
    args = parser.parse_args()

    model_dir = resolve_model_dir(args.model_dir)
    train_out = _resolve_output_path(args.train_out, model_dir / "ltr_train.parquet")
    val_out = _resolve_output_path(args.val_out, model_dir / "ltr_val.parquet")
    meta_out = _resolve_output_path(args.meta_out, model_dir / "ltr_dataset_meta.json")

    progress = StageProgress(
        total_stages=4,
        label=f"Dataset LTR {model_dir.name}",
    )

    progress.step("Carregando artefatos base")
    artifacts = load_base_artifacts(model_dir)
    message_to_rowpos = _build_message_to_rowpos(artifacts.posts_cache)
    print(f"  Catálogo disponível: {len(artifacts.posts_cache)} posts")
    print(f"  Mapeamentos message_id: {len(message_to_rowpos)}")

    rng = np.random.default_rng(args.seed)

    print()
    progress.step("Construindo dataset LTR de treino")
    train_df = _build_query_rows(
        split_name="train",
        artifacts=artifacts,
        interactions=_load_split_interactions("train"),
        features_enabled=args.features,
        negatives_per_query=args.negatives_per_query,
        hard_negative_topn=args.hard_negative_topn,
        max_queries=args.max_queries_train,
        rng=rng,
    )

    print()
    progress.step("Construindo dataset LTR de validação")
    val_df = _build_query_rows(
        split_name="val",
        artifacts=artifacts,
        interactions=_load_split_interactions("val"),
        features_enabled=args.features,
        negatives_per_query=args.negatives_per_query,
        hard_negative_topn=args.hard_negative_topn,
        max_queries=args.max_queries_val,
        rng=rng,
    )

    print()
    progress.step("Salvando datasets e metadata")
    train_out.parent.mkdir(parents=True, exist_ok=True)
    val_out.parent.mkdir(parents=True, exist_ok=True)
    train_df.to_parquet(train_out, index=False)
    val_df.to_parquet(val_out, index=False)

    categorical_maps = build_categorical_maps(artifacts.posts_cache)
    metadata = {
        "model_dir": rel_path(model_dir),
        "train_dataset": rel_path(train_out),
        "val_dataset": rel_path(val_out),
        "feature_columns": list(args.features),
        "categorical_maps": categorical_maps,
        "config": {
            "negatives_per_query": args.negatives_per_query,
            "hard_negative_topn": args.hard_negative_topn,
            "max_queries_train": args.max_queries_train,
            "max_queries_val": args.max_queries_val,
            "seed": args.seed,
        },
        "stats": {
            "train_rows": int(len(train_df)),
            "val_rows": int(len(val_df)),
            "train_queries": int(train_df["query_id"].nunique()) if not train_df.empty else 0,
            "val_queries": int(val_df["query_id"].nunique()) if not val_df.empty else 0,
            "train_positive_rate": float(train_df["label"].mean()) if not train_df.empty else 0.0,
            "val_positive_rate": float(val_df["label"].mean()) if not val_df.empty else 0.0,
        },
    }
    write_json(meta_out, metadata)

    print("\n=== Dataset LTR preparado ===")
    print(f"Treino: {train_out} ({len(train_df)} linhas)")
    print(f"Val   : {val_out} ({len(val_df)} linhas)")
    print(f"Meta  : {meta_out}")


if __name__ == "__main__":
    main()
