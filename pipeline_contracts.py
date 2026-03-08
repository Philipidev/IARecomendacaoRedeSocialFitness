from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

TIME_COLUMN_CANDIDATES = [
    "timestamp_ms",
    "event_timestamp_ms",
    "event_timestamp",
    "event_time",
    "timestamp",
    "created_at",
    "interaction_date",
    "creation_date",
]


def detect_time_column(interactions: pd.DataFrame) -> str | None:
    for col in TIME_COLUMN_CANDIDATES:
        if col in interactions.columns:
            return col
    return None


def timestamp_to_ms(value: Any) -> int | None:
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


def normalize_split_config(split_config: dict[str, Any] | None) -> dict[str, Any]:
    payload = split_config or {}
    return {
        "train": round(float(payload.get("train", 0.70)), 10),
        "val": round(float(payload.get("val", 0.15)), 10),
        "test": round(float(payload.get("test", 0.15)), 10),
        "seed": int(payload.get("seed", 42)),
    }


def split_signature(split_config: dict[str, Any] | None) -> str:
    normalized = normalize_split_config(split_config)
    raw = json.dumps(normalized, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


def load_json_optional(path: Path, default: Any = None) -> Any:
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return default


def split_signature_from_manifest_payload(payload: dict[str, Any] | None) -> str | None:
    if not isinstance(payload, dict):
        return None
    split_cfg = payload.get("split_config")
    if not isinstance(split_cfg, dict):
        return None
    return payload.get("split_signature") or split_signature(split_cfg)


def split_signature_from_manifest_file(path: Path) -> str | None:
    payload = load_json_optional(path, default={})
    return split_signature_from_manifest_payload(payload)


def split_signature_from_metadata(metadata: dict[str, Any] | None) -> str | None:
    if not isinstance(metadata, dict):
        return None
    training = metadata.get("training")
    if isinstance(training, dict):
        explicit = training.get("split_signature")
        if isinstance(explicit, str) and explicit:
            return explicit
        split_cfg = training.get("split_config")
        if isinstance(split_cfg, dict):
            return split_signature(split_cfg)
    explicit_root = metadata.get("split_signature")
    if isinstance(explicit_root, str) and explicit_root:
        return explicit_root
    split_cfg_root = metadata.get("split_config")
    if isinstance(split_cfg_root, dict):
        return split_signature(split_cfg_root)
    return None
