"""
Protocolo de avaliação offline.

Mudança importante (2025-05): o protocolo agora respeita a estratégia de split
e usa o histórico COMPLETO do usuário (train + val + test) para definir as
queries de teste. Isso evita o viés do split aleatório por post, que descartava
85% das interações como gabarito.

Estratégias suportadas (lidas do manifesto do split):

  * temporal_global  → para cada usuário com histórico ≥ 2, a referência é
    a última interação ANTES do corte de teste; o gabarito (`future_ids`)
    são as interações DEPOIS do corte. Itens já vistos antes do corte são
    excluídos do conjunto candidato.

  * leave_last_k     → para cada usuário, a referência é a última interação
    de treino+val; o gabarito são as últimas K interações (test).

  * random           → comportamento legado: usa apenas test_interactions
    como gabarito (mantido para reproduzir resultados antigos).
"""

from __future__ import annotations

import ast
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent

from dataset_context import dataset_context, dataset_context_from_metadata, manifest_path
from pipeline_contracts import (
    DEFAULT_SPLIT_STRATEGY,
    detect_time_column,
    load_json_optional,
    timestamp_to_ms,
)
from treinamento.model_utils import load_model_metadata


@dataclass
class OfflineQuery:
    user_id: int
    reference_message_id: int
    reference_timestamp_ms: int
    reference_tags: list[str]
    future_ids: set[int]
    seen_message_ids: set[int] = field(default_factory=set)


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


def load_split_strategy(splits_dir: Path) -> dict[str, Any]:
    payload = load_json_optional(manifest_path(splits_dir), default={}) or {}
    strategy = (
        payload.get("split_strategy")
        or (payload.get("split_config") or {}).get("strategy")
        or DEFAULT_SPLIT_STRATEGY
    )
    cuts = payload.get("temporal_cuts") or {}
    leave_last_k = payload.get("leave_last_k") or (
        payload.get("split_config") or {}
    ).get("leave_last_k", 1)
    return {
        "strategy": str(strategy),
        "cut_train_val_ms": cuts.get("cut_train_val_ms"),
        "cut_val_test_ms": cuts.get("cut_val_test_ms"),
        "leave_last_k": int(leave_last_k or 1),
    }


def _read_interactions_parquet(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_parquet(path)
    if df.empty:
        return df
    if "message_id" not in df.columns or "user_id" not in df.columns:
        raise ValueError(f"{path.name} precisa conter user_id e message_id.")

    tempo_col = detect_time_column(df)
    if tempo_col is None:
        raise ValueError(f"{path.name} não possui coluna temporal reconhecida.")

    df = df.copy()
    df["message_id"] = pd.to_numeric(df["message_id"], errors="coerce").astype("Int64")
    df["user_id"] = pd.to_numeric(df["user_id"], errors="coerce").astype("Int64")
    df["__ts_ms"] = df[tempo_col].apply(timestamp_to_ms)
    if df["__ts_ms"].isna().all():
        df["__ts_ms"] = np.arange(len(df), dtype=np.int64)
    df["__ts_ms"] = pd.to_numeric(df["__ts_ms"], errors="coerce")
    df = df.dropna(subset=["message_id", "user_id", "__ts_ms"]).copy()
    df["message_id"] = df["message_id"].astype("int64")
    df["user_id"] = df["user_id"].astype("int64")
    df["__ts_ms"] = df["__ts_ms"].astype("int64")
    return df


def load_split_interactions(splits_dir: Path, split_name: str) -> pd.DataFrame:
    path = splits_dir / f"{split_name}_interactions.parquet"
    if not path.exists():
        raise FileNotFoundError(
            f"Arquivo ausente: {path}. Execute primeiro python treinamento/dividir_dataset.py"
        )
    return _read_interactions_parquet(path)


def load_full_history(splits_dir: Path) -> pd.DataFrame:
    parts = []
    for split_name in ("train", "val", "test"):
        df = _read_interactions_parquet(splits_dir / f"{split_name}_interactions.parquet")
        if df.empty:
            continue
        df = df.copy()
        df["__split"] = split_name
        parts.append(df)
    if not parts:
        return pd.DataFrame()
    full = pd.concat(parts, ignore_index=True)
    return full


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


def _emit_query(
    ranker,
    user_id: int,
    reference_event: dict,
    future_events: list[dict],
    seen_ids: set[int],
    catalog_message_ids: set[int],
    message_to_row: dict[int, int],
    fallback_lookup: dict[int, int],
) -> OfflineQuery | None:
    future_ids = {
        int(item["message_id"])
        for item in future_events
        if int(item["message_id"]) in catalog_message_ids
        and int(item["message_id"]) not in seen_ids
    }
    if not future_ids:
        return None

    reference_message_id = int(reference_event["message_id"])
    ref_post = _resolve_reference_post(
        ranker, reference_message_id, message_to_row, fallback_lookup
    )
    if ref_post is None:
        return None

    reference_tags = parse_tags(ref_post.get("tags_fitness", []))
    if not reference_tags:
        return None

    reference_timestamp = timestamp_to_ms(ref_post.get("creation_date"))
    if reference_timestamp is None:
        reference_timestamp = int(reference_event["__ts_ms"])

    return OfflineQuery(
        user_id=int(user_id),
        reference_message_id=reference_message_id,
        reference_timestamp_ms=int(reference_timestamp),
        reference_tags=reference_tags,
        future_ids=future_ids,
        seen_message_ids=set(seen_ids),
    )


def _build_temporal_queries(
    ranker,
    full_history: pd.DataFrame,
    cut_val_test_ms: int | None,
    catalog_message_ids: set[int],
    message_to_row: dict[int, int],
    fallback_lookup: dict[int, int],
) -> list[OfflineQuery]:
    if cut_val_test_ms is None:
        return []

    queries: list[OfflineQuery] = []
    for user_id, group in full_history.groupby("user_id"):
        ordered = group.sort_values("__ts_ms")
        events = ordered[["message_id", "__ts_ms"]].to_dict("records")
        if len(events) < 2:
            continue

        before = [e for e in events if int(e["__ts_ms"]) <= int(cut_val_test_ms)]
        after = [e for e in events if int(e["__ts_ms"]) > int(cut_val_test_ms)]
        if not before or not after:
            continue

        reference = before[-1]
        seen_ids = {int(e["message_id"]) for e in before}
        query = _emit_query(
            ranker,
            user_id=int(user_id),
            reference_event=reference,
            future_events=after,
            seen_ids=seen_ids,
            catalog_message_ids=catalog_message_ids,
            message_to_row=message_to_row,
            fallback_lookup=fallback_lookup,
        )
        if query is not None:
            queries.append(query)
    return queries


def _build_leave_last_k_queries(
    ranker,
    full_history: pd.DataFrame,
    leave_last_k: int,
    catalog_message_ids: set[int],
    message_to_row: dict[int, int],
    fallback_lookup: dict[int, int],
) -> list[OfflineQuery]:
    leave_last_k = max(1, int(leave_last_k))
    queries: list[OfflineQuery] = []
    for user_id, group in full_history.groupby("user_id"):
        ordered = group.sort_values("__ts_ms")
        events = ordered[["message_id", "__ts_ms", "__split"]].to_dict("records")
        n = len(events)
        if n < 2:
            continue
        # Test events são os marcados como "test" (caso o split tenha sido feito
        # por leave_last_k). Se não houver, fallback: últimas K.
        test_events = [e for e in events if e.get("__split") == "test"]
        if not test_events:
            test_events = events[-leave_last_k:]
        cut_idx = events.index(test_events[0])
        if cut_idx == 0:
            continue
        reference = events[cut_idx - 1]
        seen_ids = {int(e["message_id"]) for e in events[:cut_idx]}
        query = _emit_query(
            ranker,
            user_id=int(user_id),
            reference_event=reference,
            future_events=test_events,
            seen_ids=seen_ids,
            catalog_message_ids=catalog_message_ids,
            message_to_row=message_to_row,
            fallback_lookup=fallback_lookup,
        )
        if query is not None:
            queries.append(query)
    return queries


def _build_legacy_random_queries(
    ranker,
    test_interactions: pd.DataFrame,
    catalog_message_ids: set[int],
    message_to_row: dict[int, int],
    fallback_lookup: dict[int, int],
) -> list[OfflineQuery]:
    queries: list[OfflineQuery] = []
    for user_id, group in test_interactions.groupby("user_id"):
        ordered = group.sort_values("__ts_ms")
        events = ordered[["message_id", "__ts_ms"]].to_dict("records")
        if len(events) < 2:
            continue
        for idx, event in enumerate(events[:-1]):
            future = events[idx + 1 :]
            seen_ids = {int(e["message_id"]) for e in events[: idx + 1]}
            query = _emit_query(
                ranker,
                user_id=int(user_id),
                reference_event=event,
                future_events=future,
                seen_ids=seen_ids,
                catalog_message_ids=catalog_message_ids,
                message_to_row=message_to_row,
                fallback_lookup=fallback_lookup,
            )
            if query is not None:
                queries.append(query)
    return queries


def build_future_queries(
    ranker,
    interactions: pd.DataFrame,
    output_dir: Path,
    *,
    splits_dir: Path | None = None,
) -> list[OfflineQuery]:
    """
    Constrói queries de teste respeitando a estratégia de split.

    `interactions` é o conjunto de teste (compatibilidade com chamadas antigas).
    Se `splits_dir` for fornecido e o manifesto indicar split temporal, o
    histórico completo é usado para evitar fragmentação do gabarito.
    """
    message_to_row, fallback_lookup = build_catalog_lookup(ranker, output_dir)
    catalog_message_ids = set(message_to_row)

    strategy_info = (
        load_split_strategy(splits_dir) if splits_dir is not None else {"strategy": "random"}
    )
    strategy = strategy_info.get("strategy", "random")

    if strategy == "temporal_global" and splits_dir is not None:
        full_history = load_full_history(splits_dir)
        if full_history.empty:
            return []
        return _build_temporal_queries(
            ranker,
            full_history=full_history,
            cut_val_test_ms=strategy_info.get("cut_val_test_ms"),
            catalog_message_ids=catalog_message_ids,
            message_to_row=message_to_row,
            fallback_lookup=fallback_lookup,
        )

    if strategy == "leave_last_k" and splits_dir is not None:
        full_history = load_full_history(splits_dir)
        if full_history.empty:
            return []
        return _build_leave_last_k_queries(
            ranker,
            full_history=full_history,
            leave_last_k=strategy_info.get("leave_last_k", 1),
            catalog_message_ids=catalog_message_ids,
            message_to_row=message_to_row,
            fallback_lookup=fallback_lookup,
        )

    return _build_legacy_random_queries(
        ranker,
        test_interactions=interactions,
        catalog_message_ids=catalog_message_ids,
        message_to_row=message_to_row,
        fallback_lookup=fallback_lookup,
    )
