from __future__ import annotations

import ast
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent

from dataset_context import dataset_context, dataset_context_from_metadata
from pipeline_contracts import detect_time_column, timestamp_to_ms
from treinamento.model_utils import load_model_metadata


@dataclass
class OfflineQuery:
    user_id: int
    reference_message_id: int
    reference_timestamp_ms: int
    reference_tags: list[str]
    future_ids: set[int]


def parse_tags(value: Any) -> list[str]:
    if isinstance(value, (list, np.ndarray)):
        return [str(t) for t in value]
    if isinstance(value, str):
        try:
            parsed = ast.literal_eval(value)
            return [str(t) for t in parsed] if isinstance(parsed, list) else [value]
        except Exception:
            return [value]
    return []


def resolve_dataset_dirs(
    model_dir: Path,
    dataset_key: str | None,
    splits_dir: str | None,
    output_dir: str | None,
) -> tuple[Path, Path]:
    metadata = load_model_metadata(model_dir)
    context = dataset_context_from_metadata(metadata)
    if context is None:
        context = dataset_context(dataset_key=dataset_key)

    resolved_splits = Path(splits_dir) if splits_dir else context.splits_dir
    if not resolved_splits.is_absolute():
        resolved_splits = (ROOT / resolved_splits).resolve()

    resolved_output = Path(output_dir) if output_dir else context.output_dir
    if not resolved_output.is_absolute():
        resolved_output = (ROOT / resolved_output).resolve()

    return resolved_splits, resolved_output


def load_split_interactions(splits_dir: Path, split_name: str) -> pd.DataFrame:
    path = splits_dir / f"{split_name}_interactions.parquet"
    if not path.exists():
        raise FileNotFoundError(
            f"Arquivo ausente: {path}. Execute primeiro python treinamento/dividir_dataset.py"
        )

    df = pd.read_parquet(path)
    if "message_id" not in df.columns or "user_id" not in df.columns:
        raise ValueError(
            f"{path.name} precisa conter as colunas user_id e message_id."
        )

    tempo_col = detect_time_column(df)
    if tempo_col is None:
        raise ValueError(f"{path.name} não possui coluna temporal reconhecida.")

    df = df.copy()
    df["message_id"] = pd.to_numeric(df["message_id"], errors="coerce").astype("Int64")
    df["user_id"] = pd.to_numeric(df["user_id"], errors="coerce").astype("Int64")
    df["__ts_ms"] = df[tempo_col].apply(timestamp_to_ms)
    if df["__ts_ms"].isna().all():
        df["__ts_ms"] = np.arange(len(df), dtype=np.int64)
    df = df.dropna(subset=["message_id", "user_id"]).copy()
    df["message_id"] = df["message_id"].astype("int64")
    df["user_id"] = df["user_id"].astype("int64")
    return df


def build_catalog_lookup(
    ranker,
    output_dir: Path,
) -> tuple[dict[int, int], dict[int, int]]:
    posts = ranker.artifacts.posts_cache
    message_to_row: dict[int, int] = {}
    if "_message_id" in posts.columns:
        values = pd.to_numeric(posts["_message_id"], errors="coerce")
        for row_pos, value in enumerate(values):
            if pd.notna(value):
                message_to_row[int(value)] = row_pos

    fallback_lookup: dict[int, int] = {}
    msgs_path = output_dir / "messages_fitness.parquet"
    if msgs_path.exists():
        msgs_df = pd.read_parquet(msgs_path)
        if "message_id" in msgs_df.columns:
            values = pd.to_numeric(msgs_df["message_id"], errors="coerce")
            fallback_lookup = {
                int(value): int(idx)
                for idx, value in enumerate(values)
                if pd.notna(value)
            }

    return message_to_row, fallback_lookup


def _resolve_reference_post(
    ranker,
    message_id: int,
    message_to_row: dict[int, int],
    fallback_lookup: dict[int, int],
):
    posts_cache = ranker.artifacts.posts_cache
    if message_id in message_to_row:
        return posts_cache.iloc[message_to_row[message_id]]

    fallback_idx = fallback_lookup.get(message_id)
    if fallback_idx is None or fallback_idx not in posts_cache.index:
        return None
    return posts_cache.loc[fallback_idx]


def build_future_queries(
    ranker,
    interactions: pd.DataFrame,
    output_dir: Path,
) -> list[OfflineQuery]:
    message_to_row, fallback_lookup = build_catalog_lookup(ranker, output_dir)
    catalog_message_ids = set(message_to_row)
    queries: list[OfflineQuery] = []

    for user_id, group in interactions.groupby("user_id"):
        ordered = group.sort_values("__ts_ms")
        events = ordered[["message_id", "__ts_ms"]].to_dict("records")
        if len(events) < 2:
            continue

        for idx, event in enumerate(events[:-1]):
            reference_message_id = int(event["message_id"])
            future_ids = {
                int(item["message_id"])
                for item in events[idx + 1 :]
                if int(item["message_id"]) in catalog_message_ids
            }
            if not future_ids:
                continue

            ref_post = _resolve_reference_post(
                ranker,
                reference_message_id,
                message_to_row,
                fallback_lookup,
            )
            if ref_post is None:
                continue

            reference_tags = parse_tags(ref_post.get("tags_fitness", []))
            if not reference_tags:
                continue

            reference_timestamp = timestamp_to_ms(ref_post.get("creation_date"))
            if reference_timestamp is None:
                reference_timestamp = int(event["__ts_ms"])

            queries.append(
                OfflineQuery(
                    user_id=int(user_id),
                    reference_message_id=reference_message_id,
                    reference_timestamp_ms=int(reference_timestamp),
                    reference_tags=reference_tags,
                    future_ids=future_ids,
                )
            )

    return queries
