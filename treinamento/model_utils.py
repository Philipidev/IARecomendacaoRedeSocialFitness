from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
DADOS_DIR = ROOT / "treinamento" / "dados"
DEFAULT_MODEL_DIR = ROOT / "treinamento" / "modelo"
MODELOS_DIR = ROOT / "treinamento" / "modelos"


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def rel_path(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT.resolve()).as_posix()
    except ValueError:
        return str(path.resolve())


def resolve_model_dir(model_dir: str | Path | None = None) -> Path:
    if model_dir is None:
        return DEFAULT_MODEL_DIR
    path = Path(model_dir)
    return path if path.is_absolute() else ROOT / path


def ensure_model_dir(model_dir: str | Path | None = None) -> Path:
    path = resolve_model_dir(model_dir)
    path.mkdir(parents=True, exist_ok=True)
    return path


def metadata_path(model_dir: str | Path | None = None) -> Path:
    return resolve_model_dir(model_dir) / "metadata.json"


def load_json_optional(path: Path, default: Any = None) -> Any:
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return default


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def load_model_metadata(model_dir: str | Path | None = None) -> dict[str, Any]:
    payload = load_json_optional(metadata_path(model_dir), default={})
    return payload if isinstance(payload, dict) else {}


def _deep_merge(base: dict[str, Any], extra: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in extra.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def merge_model_metadata(model_dir: str | Path | None, payload: dict[str, Any]) -> Path:
    path = metadata_path(model_dir)
    current = load_model_metadata(model_dir)
    payload = dict(payload)
    payload.setdefault("updated_at", now_iso())
    merged = _deep_merge(current, payload)
    write_json(path, merged)
    return path


def infer_model_family(model_dir: str | Path | None = None) -> str:
    path = resolve_model_dir(model_dir)
    metadata = load_model_metadata(path)
    family = metadata.get("family")
    if isinstance(family, str) and family:
        return family
    if (path / "ltr_model.txt").exists():
        return "ltr_lightgbm"
    return "baseline_hibrido"


def model_id_from_dir(model_dir: str | Path | None = None) -> str:
    path = resolve_model_dir(model_dir)
    metadata = load_model_metadata(path)
    model_id = metadata.get("id")
    if isinstance(model_id, str) and model_id:
        return model_id
    return path.name
