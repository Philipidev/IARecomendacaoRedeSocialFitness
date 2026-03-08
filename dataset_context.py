from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent
EXTRACAO_DIR = ROOT / "extracao_filtragem"
DATASET_ARCHIVES_DIR = EXTRACAO_DIR / "dataset"
EXTRACTION_ROOT_DIR = EXTRACAO_DIR / "ldbc_snb"
OUTPUT_ROOT_DIR = EXTRACAO_DIR / "output"
TREINAMENTO_DIR = ROOT / "treinamento"
DADOS_ROOT_DIR = TREINAMENTO_DIR / "dados"
MODELOS_ROOT_DIR = TREINAMENTO_DIR / "modelos"
RESULTADOS_ROOT_DIR = ROOT / "avaliacao" / "resultados"

LEGACY_DATASET_KEY = "legacy_default"
MANIFEST_FILENAME = "dataset_manifest.json"


def now_iso() -> str:
    from datetime import datetime, timezone

    return datetime.now(timezone.utc).isoformat()


def rel_path(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT.resolve()).as_posix()
    except ValueError:
        return str(path.resolve())


def abs_path(path_str: str | None) -> Path | None:
    if not path_str:
        return None
    path = Path(path_str)
    return path if path.is_absolute() else ROOT / path


def detect_scale_factor(filename: str) -> str | None:
    match = re.search(r"sf\d+(?:\.\d+)?", filename)
    return match.group(0) if match else None


def strip_archive_suffix(filename: str) -> str:
    lowered = filename.lower()
    for suffix in [".tar.zst", ".tar.gz", ".tar.xz", ".tar.bz2", ".tar", ".zst", ".gz"]:
        if lowered.endswith(suffix):
            return filename[: -len(suffix)]
    return Path(filename).stem


def dataset_key_from_path(dataset_path: str | Path) -> str:
    return strip_archive_suffix(Path(dataset_path).name)


def resolve_dataset_key(
    dataset_key: str | None = None,
    dataset_path: str | Path | None = None,
) -> str | None:
    if dataset_key:
        return str(dataset_key).strip()
    if dataset_path:
        return dataset_key_from_path(dataset_path)
    return None


@dataclass(frozen=True)
class DatasetContext:
    dataset_key: str | None
    dataset_path: str | None
    scale_factor: str | None
    extraction_dir: Path
    output_dir: Path
    dados_dir: Path
    splits_dir: Path
    models_dir: Path
    results_dir: Path
    is_legacy: bool

    def to_metadata(self) -> dict[str, Any]:
        payload = asdict(self)
        for key in [
            "extraction_dir",
            "output_dir",
            "dados_dir",
            "splits_dir",
            "models_dir",
            "results_dir",
        ]:
            payload[key] = rel_path(payload[key])
        return payload


def dataset_context(
    *,
    dataset_key: str | None = None,
    dataset_path: str | Path | None = None,
    scale_factor: str | None = None,
    use_legacy: bool = False,
) -> DatasetContext:
    resolved_path = abs_path(str(dataset_path)) if dataset_path is not None else None
    resolved_key = resolve_dataset_key(dataset_key=dataset_key, dataset_path=dataset_path)
    if use_legacy or not resolved_key:
        return DatasetContext(
            dataset_key=LEGACY_DATASET_KEY if use_legacy else None,
            dataset_path=rel_path(resolved_path) if resolved_path else None,
            scale_factor=scale_factor
            or (detect_scale_factor(resolved_path.name) if resolved_path else None),
            extraction_dir=EXTRACTION_ROOT_DIR,
            output_dir=OUTPUT_ROOT_DIR,
            dados_dir=DADOS_ROOT_DIR,
            splits_dir=DADOS_ROOT_DIR / "splits",
            models_dir=MODELOS_ROOT_DIR,
            results_dir=RESULTADOS_ROOT_DIR,
            is_legacy=True,
        )

    return DatasetContext(
        dataset_key=resolved_key,
        dataset_path=rel_path(resolved_path) if resolved_path else None,
        scale_factor=scale_factor
        or (detect_scale_factor(resolved_path.name) if resolved_path else None),
        extraction_dir=EXTRACTION_ROOT_DIR / resolved_key,
        output_dir=OUTPUT_ROOT_DIR / resolved_key,
        dados_dir=DADOS_ROOT_DIR / resolved_key,
        splits_dir=(DADOS_ROOT_DIR / resolved_key / "splits"),
        models_dir=MODELOS_ROOT_DIR / resolved_key,
        results_dir=RESULTADOS_ROOT_DIR / resolved_key,
        is_legacy=False,
    )


def ensure_context_dirs(context: DatasetContext) -> None:
    DATASET_ARCHIVES_DIR.mkdir(parents=True, exist_ok=True)
    context.extraction_dir.mkdir(parents=True, exist_ok=True)
    context.output_dir.mkdir(parents=True, exist_ok=True)
    context.dados_dir.mkdir(parents=True, exist_ok=True)
    context.splits_dir.mkdir(parents=True, exist_ok=True)
    context.models_dir.mkdir(parents=True, exist_ok=True)
    context.results_dir.mkdir(parents=True, exist_ok=True)


def manifest_path(base_dir: Path) -> Path:
    return base_dir / MANIFEST_FILENAME


def load_json_optional(path: Path, default: Any = None) -> Any:
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return default


def load_manifest(base_dir: Path) -> dict[str, Any]:
    payload = load_json_optional(manifest_path(base_dir), default={})
    return payload if isinstance(payload, dict) else {}


def write_manifest(base_dir: Path, payload: dict[str, Any]) -> Path:
    base_dir.mkdir(parents=True, exist_ok=True)
    path = manifest_path(base_dir)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def build_stage_manifest(
    *,
    stage: str,
    context: DatasetContext,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "stage": stage,
        "generated_at": now_iso(),
        "dataset": context.to_metadata(),
    }
    if extra:
        payload.update(extra)
    return payload


def dataset_context_from_metadata(metadata: dict[str, Any]) -> DatasetContext | None:
    dataset = metadata.get("dataset")
    if not isinstance(dataset, dict):
        return None

    dataset_key = dataset.get("dataset_key")
    dataset_path = dataset.get("dataset_path")
    scale_factor = dataset.get("scale_factor")
    use_legacy = bool(dataset.get("is_legacy", False))
    context = dataset_context(
        dataset_key=dataset_key,
        dataset_path=dataset_path,
        scale_factor=scale_factor,
        use_legacy=use_legacy,
    )

    def _path_or_default(key: str, default: Path) -> Path:
        return abs_path(dataset.get(key)) or default

    return DatasetContext(
        dataset_key=context.dataset_key,
        dataset_path=context.dataset_path,
        scale_factor=context.scale_factor,
        extraction_dir=_path_or_default("extraction_dir", context.extraction_dir),
        output_dir=_path_or_default("output_dir", context.output_dir),
        dados_dir=_path_or_default("dados_dir", context.dados_dir),
        splits_dir=_path_or_default("splits_dir", context.splits_dir),
        models_dir=_path_or_default("models_dir", context.models_dir),
        results_dir=_path_or_default("results_dir", context.results_dir),
        is_legacy=use_legacy or context.is_legacy,
    )


def default_model_dir_for_dataset(dataset_key: str | None = None) -> Path:
    context = dataset_context(dataset_key=dataset_key)
    if context.is_legacy:
        return ROOT / "treinamento" / "modelo"
    return context.models_dir / "modelo_padrao"
