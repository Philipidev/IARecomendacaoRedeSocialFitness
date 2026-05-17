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


_MS_PER_SECOND = 1_000
_MS_PER_MICROSECOND = 1 / 1_000


def _ms_from_numeric(value: float | int) -> int:
    """Aplica heurística por magnitude para inferir a unidade de um timestamp numérico.

    LDBC SNB com LongDateFormatter exporta milissegundos desde 1970-01-01 (long Java),
    valores tipicamente >= 1e12. Datas pós-2001 em segundos ficam em ~1e9. Microssegundos
    ficam ~1e15 e nanossegundos ~1e18. Sem essa heurística, valores em segundos seriam
    interpretados como milissegundos (gerando datas no ano ~1970) e vice-versa.
    """
    abs_value = abs(float(value))
    if abs_value >= 1e17:
        return int(value // 1_000_000)
    if abs_value >= 1e14:
        return int(value // 1_000)
    if abs_value >= 1e10:
        return int(value)
    return int(value * _MS_PER_SECOND)


def timestamp_to_ms(value: Any) -> int | None:
    """Converte um valor heterogêneo em milissegundos UTC desde 1970-01-01.

    Aceita ``int``/``float`` numéricos (com inferência de unidade), strings que sejam
    apenas dígitos (tratadas como long em ms — formato LDBC SNB ``LongDateFormatter``)
    e strings ISO 8601 / similares (delegadas a ``pandas.to_datetime``).
    Retorna ``None`` apenas quando o valor é nulo/inválido.
    """
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass

    if isinstance(value, bool):
        return None
    if isinstance(value, (int, np.integer)):
        return _ms_from_numeric(int(value))
    if isinstance(value, (float, np.floating)):
        if not np.isfinite(value):
            return None
        return _ms_from_numeric(float(value))

    if isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        sign = -1 if text.startswith("-") else 1
        digits = text[1:] if text[0] in "+-" else text
        if digits.isdigit():
            return _ms_from_numeric(sign * int(digits))

    try:
        dt = pd.to_datetime(value, utc=True, errors="raise")
    except Exception:
        try:
            dt = pd.to_datetime(str(value), utc=True, errors="raise")
        except Exception:
            return None

    if dt is pd.NaT or pd.isna(dt):
        return None
    try:
        return int(dt.value // 1_000_000)
    except AttributeError:
        ts = pd.Timestamp(dt)
        return int(ts.value // 1_000_000)


def timestamps_series_to_ms(series: "pd.Series") -> "pd.Series":
    """Converte uma Series de timestamps heterogêneos para inteiros em ms (Int64).

    Versão vetorizada de ``timestamp_to_ms`` para ser usada na pipeline de splits e
    avaliação. Lida com colunas em ``int64``/``float64`` (LongDateFormatter),
    ``datetime64[ns]`` (pyarrow timestamp) e strings (delegadas para ``pd.to_datetime``).
    Retorna sempre ``Int64`` (com ``pd.NA`` para entradas inválidas).
    """
    if series is None or len(series) == 0:
        return pd.Series([], dtype="Int64")

    out = pd.Series(pd.NA, index=series.index, dtype="Int64")

    if pd.api.types.is_datetime64_any_dtype(series):
        dt = pd.to_datetime(series, utc=True, errors="coerce")
        as_int = dt.astype("int64", errors="ignore") // 1_000_000
        valid = dt.notna()
        out.loc[valid] = as_int.loc[valid].astype("Int64")
        return out

    numeric = pd.to_numeric(series, errors="coerce")
    numeric_mask = numeric.notna()
    if numeric_mask.any():
        magnitudes = numeric.loc[numeric_mask].astype(float).abs()
        scaled = numeric.loc[numeric_mask].astype(float)
        thresholds = [1e17, 1e14, 1e10]
        divisors = [1_000_000.0, 1_000.0, 1.0]
        for threshold, divisor in zip(thresholds, divisors):
            mask_band = magnitudes >= threshold
            if mask_band.any():
                idx = scaled.index[mask_band]
                out.loc[idx] = (scaled.loc[idx] // divisor).astype("Int64")
                magnitudes = magnitudes.loc[~mask_band]
                scaled = scaled.loc[~mask_band]
        if not scaled.empty:
            idx = scaled.index
            out.loc[idx] = (scaled * _MS_PER_SECOND).astype("Int64")

    remaining_mask = out.isna() & series.notna() & ~numeric_mask
    if remaining_mask.any():
        parsed = pd.to_datetime(series.loc[remaining_mask], utc=True, errors="coerce")
        valid = parsed.notna()
        if valid.any():
            ms_values = (parsed.loc[valid].astype("int64") // 1_000_000)
            out.loc[ms_values.index] = ms_values.astype("Int64")

    return out


VALID_SPLIT_STRATEGIES = ("random", "temporal_global", "leave_last_k")
DEFAULT_SPLIT_STRATEGY = "temporal_global"


def normalize_split_config(split_config: dict[str, Any] | None) -> dict[str, Any]:
    payload = split_config or {}
    strategy = str(payload.get("strategy") or DEFAULT_SPLIT_STRATEGY).strip()
    if strategy not in VALID_SPLIT_STRATEGIES:
        strategy = DEFAULT_SPLIT_STRATEGY
    leave_last_k = int(payload.get("leave_last_k", 1) or 1)
    return {
        "train": round(float(payload.get("train", 0.70)), 10),
        "val": round(float(payload.get("val", 0.15)), 10),
        "test": round(float(payload.get("test", 0.15)), 10),
        "seed": int(payload.get("seed", 42)),
        "strategy": strategy,
        "leave_last_k": max(1, leave_last_k),
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
