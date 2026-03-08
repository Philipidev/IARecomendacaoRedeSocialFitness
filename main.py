#!/usr/bin/env python3
"""
Orquestrador interativo do pipeline de recomendacao fitness.

Centraliza download/seleção de dataset, extracao, treinamento e avaliacao,
mantendo um arquivo de estado na raiz do projeto para reaproveitar o contexto
entre execucoes.
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from dataset_context import (
    DatasetContext,
    dataset_context,
    default_model_dir_for_dataset,
    load_manifest,
)
from pipeline_contracts import split_signature_from_manifest_payload, split_signature_from_metadata
from treinamento.model_utils import merge_model_metadata

ROOT = Path(__file__).resolve().parent
STATE_PATH = ROOT / ".pipeline_state.json"

DATASET_DIR = ROOT / "extracao_filtragem" / "dataset"
OUTPUT_DIR = ROOT / "extracao_filtragem" / "output"
DADOS_DIR = ROOT / "treinamento" / "dados"
SPLITS_DIR = DADOS_DIR / "splits"
MODELO_DIR = ROOT / "treinamento" / "modelo"
AVALIACAO_DIR = ROOT / "avaliacao"
RESULTADOS_DIR = AVALIACAO_DIR / "resultados"
MODELOS_DIR = ROOT / "treinamento" / "modelos"
TCC_CONFIG_PATH = ROOT / "casos_uso_tcc.json"

DOWNLOAD_SCRIPT = ROOT / "extracao_filtragem" / "download_dataset.py"
PIPELINE_SCRIPT = ROOT / "extracao_filtragem" / "pipeline.py"
PREPARAR_SCRIPT = ROOT / "treinamento" / "preparacao_dados.py"
DIVIDIR_SCRIPT = ROOT / "treinamento" / "dividir_dataset.py"
TREINAR_SCRIPT = ROOT / "treinamento" / "treinar.py"
PREPARAR_LTR_SCRIPT = ROOT / "treinamento" / "preparar_dataset_ltr.py"
TREINAR_LTR_SCRIPT = ROOT / "treinamento" / "treinar_ltr.py"
AVALIAR_MODELO_SCRIPT = ROOT / "avaliacao" / "avaliar_modelo.py"
AVALIAR_POPULARIDADE_SCRIPT = ROOT / "avaliacao" / "avaliar_popularidade.py"
AVALIACAO_MANUAL_SCRIPT = ROOT / "avaliacao" / "avaliacao_manual.py"
OTIMIZAR_PESOS_SCRIPT = ROOT / "avaliacao" / "otimizar_pesos.py"
BENCHMARK_TCC_SCRIPT = ROOT / "avaliacao" / "benchmark_modelos.py"

STATE_VERSION = 2

DEFAULT_SPLIT_CONFIG = {
    "train": 0.70,
    "val": 0.15,
    "test": 0.15,
    "seed": 42,
}

DOWNLOAD_OPTIONS = [
    {
        "scale_factor": "sf0.1",
        "size": "~18 MB",
        "descricao": "Desenvolvimento e testes",
        "filename": "social_network-sf0.1-CsvBasic-LongDateFormatter.tar.zst",
    },
    {
        "scale_factor": "sf0.3",
        "size": "~50 MB",
        "descricao": "Validacao local",
        "filename": "social_network-sf0.3-CsvBasic-LongDateFormatter.tar.zst",
    },
    {
        "scale_factor": "sf1",
        "size": "~160 MB",
        "descricao": "Experimentos medios",
        "filename": "social_network-sf1-CsvBasic-LongDateFormatter.tar.zst",
    },
    {
        "scale_factor": "sf3",
        "size": "~500 MB",
        "descricao": "Treinamento real",
        "filename": "social_network-sf3-CsvBasic-LongDateFormatter.tar.zst",
    },
    {
        "scale_factor": "sf10",
        "size": "~1.7 GB",
        "descricao": "Producao",
        "filename": "social_network-sf10-CsvBasic-LongDateFormatter.tar.zst",
    },
    {
        "scale_factor": "sf30",
        "size": "~20 GB",
        "descricao": "Producao em larga escala",
        "filename": "social_network-sf30-CsvBasic-LongDateFormatter.tar.zst",
    },
]

MESSAGES_FITNESS_FILE = "messages_fitness.parquet"
PESOS_OTIMOS_FILENAME = "pesos_otimos.json"
PESOS_EXPERIMENTOS_FILENAME = "pesos_experimentos.csv"
INVALID_OPTION_MESSAGE = "Opcao invalida. Tente novamente."

EXTRACAO_OUTPUTS = [
    MESSAGES_FITNESS_FILE,
    "tags_fitness.parquet",
    "tag_cooccurrence.parquet",
    "interactions_fitness.parquet",
    "user_interests_fitness.parquet",
    "user_social_graph.parquet",
]

TREINAMENTO_DADOS_OUTPUTS = [
    "posts_metadata.parquet",
    "interacoes_por_tag.parquet",
    "social_scores.parquet",
    "user_tag_profile.parquet",
    "tag_lista.txt",
    "event_type_lista.txt",
    "language_lista.txt",
    "message_type_lista.txt",
    "user_id_lista.txt",
    "tag_cooccurrence_pares_lista.txt",
]

TREINAMENTO_SPLITS_OUTPUTS = [
    "train_posts.parquet",
    "val_posts.parquet",
    "test_posts.parquet",
    "train_interactions.parquet",
    "val_interactions.parquet",
    "test_interactions.parquet",
    "train_tag_cooccurrence.parquet",
    "train_social_scores.parquet",
]

TREINAMENTO_MODELO_REQUIRED = [
    "vectorizer.pkl",
    "post_matrix.npy",
    "tag_cooccurrence_map.pkl",
    "popularidade.npy",
    "social_scores.npy",
    "posts_cache.parquet",
]

TREINAMENTO_MODELO_OPTIONAL = [
    PESOS_OTIMOS_FILENAME,
]

TREINAMENTO_LTR_REQUIRED = [
    "ltr_model.txt",
    "ltr_feature_schema.json",
]

AVALIACAO_OFFLINE_OUTPUTS = [
    "metricas_resumo.json",
    "metricas_ranking_por_k.csv",
    "queries_avaliadas.csv",
    "resumo_avaliacao.md",
]

BENCHMARK_TCC_OUTPUTS = [
    "benchmark_modelos.csv",
    "benchmark_modelos.md",
    "benchmark_modelos.json",
]


def now_iso() -> str:
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


def format_size(num_bytes: int) -> str:
    if num_bytes < 1024:
        return f"{num_bytes} B"
    units = ["KB", "MB", "GB", "TB"]
    value = float(num_bytes)
    for unit in units:
        value /= 1024.0
        if value < 1024 or unit == units[-1]:
            return f"{value:.1f} {unit}"
    return f"{num_bytes} B"


def detect_scale_factor(filename: str) -> str | None:
    match = re.search(r"sf\d+(?:\.\d+)?", filename)
    return match.group(0) if match else None


def load_json_optional(path: Path, default: Any = None) -> Any:
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return default


def infer_model_family_from_disk(model_dir: Path, fallback: str = "baseline_hibrido") -> str:
    metadata = load_json_optional(model_dir / "metadata.json", default={})
    family = metadata.get("family") if isinstance(metadata, dict) else None
    if isinstance(family, str) and family:
        return family
    if (model_dir / "ltr_model.txt").exists():
        return "ltr_lightgbm"
    return fallback


def find_download_option(scale_factor: str | None) -> dict[str, Any] | None:
    if not scale_factor:
        return None
    for option in DOWNLOAD_OPTIONS:
        if option["scale_factor"] == scale_factor:
            return option
    return None


def default_model_target(dataset_key: str | None = None) -> dict[str, Any]:
    model_dir = default_model_dir_for_dataset(dataset_key)
    return {
        "type": "modelo_padrao",
        "model_id": "modelo_padrao",
        "model_dir": rel_path(model_dir),
        "family": infer_model_family_from_disk(model_dir),
        "label": "Modelo padrão",
    }


def default_benchmark_target() -> dict[str, Any]:
    return {
        "scope": "all",
        "model_ids": [],
    }


def load_tcc_config_safe() -> tuple[dict[str, Any] | None, str | None]:
    if not TCC_CONFIG_PATH.exists():
        return None, f"Arquivo ausente: {rel_path(TCC_CONFIG_PATH)}"
    payload = load_json_optional(TCC_CONFIG_PATH, default=None)
    if not isinstance(payload, dict):
        return None, "casos_uso_tcc.json inválido: raiz deve ser um objeto JSON."
    modelos = payload.get("modelos")
    if not isinstance(modelos, list):
        return None, "casos_uso_tcc.json inválido: 'modelos' deve ser uma lista."
    return payload, None


def tcc_models_from_config(
    config: dict[str, Any] | None,
    *,
    enabled_only: bool = False,
) -> list[dict[str, Any]]:
    if not isinstance(config, dict):
        return []
    modelos = config.get("modelos")
    if not isinstance(modelos, list):
        return []
    output: list[dict[str, Any]] = []
    for item in modelos:
        if not isinstance(item, dict) or "id" not in item:
            continue
        if enabled_only and not bool(item.get("enabled", True)):
            continue
        output.append(item)
    return output


def tcc_model_map(config: dict[str, Any] | None) -> dict[str, dict[str, Any]]:
    return {
        str(item["id"]): item
        for item in tcc_models_from_config(config, enabled_only=False)
    }


def make_experiment_model_target(
    model_cfg: dict[str, Any],
    *,
    dataset_key: str | None = None,
    selected_at: str | None = None,
) -> dict[str, Any]:
    experiment_id = str(model_cfg["id"])
    model_dir = dataset_context(dataset_key=dataset_key).models_dir / experiment_id
    payload = {
        "type": "experimento_tcc",
        "experiment_id": experiment_id,
        "model_dir": rel_path(model_dir),
        "family": str(model_cfg.get("family", "baseline_hibrido")),
        "label": f"{experiment_id} ({model_cfg.get('family', 'baseline_hibrido')})",
        "descricao": str(model_cfg.get("descricao", "")),
        "enabled": bool(model_cfg.get("enabled", True)),
    }
    if selected_at:
        payload["selected_at"] = selected_at
    return payload


def normalize_model_target(
    target: Any,
    config: dict[str, Any] | None,
    dataset_key: str | None = None,
) -> dict[str, Any]:
    selected_at = target.get("selected_at") if isinstance(target, dict) else None
    if isinstance(target, dict) and target.get("type") == "experimento_tcc":
        experiment_id = str(target.get("experiment_id", "")).strip()
        if experiment_id:
            model_cfg = tcc_model_map(config).get(experiment_id)
            if model_cfg is not None:
                return make_experiment_model_target(
                    model_cfg,
                    dataset_key=dataset_key,
                    selected_at=selected_at,
                )

            default_model_dir = dataset_context(dataset_key=dataset_key).models_dir / experiment_id
            model_dir = abs_path(target.get("model_dir")) or default_model_dir
            payload = {
                "type": "experimento_tcc",
                "experiment_id": experiment_id,
                "model_dir": rel_path(model_dir),
                "family": str(
                    target.get("family")
                    or infer_model_family_from_disk(model_dir)
                ),
                "label": str(
                    target.get("label")
                    or f"{experiment_id} (fora do casos_uso_tcc.json)"
                ),
                "descricao": str(target.get("descricao", "")),
                "enabled": bool(target.get("enabled", False)),
            }
            if selected_at:
                payload["selected_at"] = selected_at
            return payload

    payload = default_model_target(dataset_key=dataset_key)
    if selected_at:
        payload["selected_at"] = selected_at
    return payload


def normalize_benchmark_target(
    target: Any,
    config: dict[str, Any] | None,
) -> dict[str, Any]:
    if not isinstance(target, dict):
        return default_benchmark_target()

    enabled_ids = {
        str(model_cfg["id"])
        for model_cfg in tcc_models_from_config(config, enabled_only=True)
    }
    scope = "subset" if target.get("scope") == "subset" else "all"
    model_ids = []
    for model_id in target.get("model_ids", []):
        model_id_str = str(model_id)
        if model_id_str in enabled_ids and model_id_str not in model_ids:
            model_ids.append(model_id_str)

    if scope == "subset" and not model_ids:
        return default_benchmark_target()
    return {
        "scope": scope,
        "model_ids": model_ids,
    }


def resolve_model_target_dir(target: dict[str, Any]) -> Path:
    model_dir = abs_path(target.get("model_dir"))
    if model_dir is not None:
        return model_dir
    return default_model_dir_for_dataset()


def build_model_dir_status(
    model_dir: Path,
    family: str | None = None,
    *,
    selected_dataset: dict[str, Any] | None = None,
    split_manifest: dict[str, Any] | None = None,
) -> dict[str, Any]:
    family_norm = infer_model_family_from_disk(model_dir, fallback=family or "baseline_hibrido")
    required_paths = [model_dir / name for name in TREINAMENTO_MODELO_REQUIRED]
    if family_norm == "ltr_lightgbm":
        required_paths.extend(model_dir / name for name in TREINAMENTO_LTR_REQUIRED)
    metadata = load_json_optional(model_dir / "metadata.json", default={})
    dataset_meta = metadata.get("dataset", {}) if isinstance(metadata, dict) else {}
    dataset_match = manifest_matches_selected_dataset(
        {"dataset": dataset_meta} if isinstance(dataset_meta, dict) else {},
        selected_dataset,
    )
    split_match = split_signature_matches_manifest(
        metadata if isinstance(metadata, dict) else None,
        split_manifest if isinstance(split_manifest, dict) else None,
    )
    required_status = build_file_status(required_paths)
    required_status["dataset_match"] = dataset_match
    required_status["split_match"] = split_match
    required_status["stale_split"] = (split_match is False)
    required_status["ready"] = (
        required_status["ready"]
        and dataset_match
        and split_match is not False
    )
    return {
        "model_dir": rel_path(model_dir),
        "family": family_norm,
        "required": required_status,
        "optional": build_file_status([model_dir / name for name in TREINAMENTO_MODELO_OPTIONAL]),
        "metadata_exists": (model_dir / "metadata.json").exists(),
        "dataset": dataset_meta if isinstance(dataset_meta, dict) else {},
    }


def model_target_label(target: dict[str, Any]) -> str:
    target_type = str(target.get("type", "modelo_padrao"))
    if target_type == "experimento_tcc":
        experiment_id = target.get("experiment_id", "sem_id")
        family = target.get("family", "desconhecida")
        return f"{experiment_id} [{family}]"
    family = target.get("family", "baseline_hibrido")
    return f"modelo padrão [{family}]"


def selected_model_target_label(state: dict[str, Any]) -> str:
    target = state.get("selected_model_target")
    if not isinstance(target, dict):
        target = default_model_target()
    return model_target_label(target)


def benchmark_target_label(state: dict[str, Any]) -> str:
    target = state.get("selected_benchmark")
    if not isinstance(target, dict) or target.get("scope", "all") != "subset":
        return "todos os modelos habilitados"
    model_ids = target.get("model_ids", [])
    if not model_ids:
        return "todos os modelos habilitados"
    return ", ".join(str(model_id) for model_id in model_ids)


def update_selected_dataset(
    state: dict[str, Any],
    dataset_path: Path,
    *,
    scale_factor: str | None,
    source: str,
) -> None:
    context = dataset_context(dataset_path=dataset_path, scale_factor=scale_factor)
    state["selected_dataset"] = {
        "path": rel_path(dataset_path),
        "dataset_key": context.dataset_key,
        "scale_factor": context.scale_factor,
        "source": source,
        "selected_at": now_iso(),
        "exists": dataset_path.exists(),
    }


def current_dataset_context(state: dict[str, Any], *, legacy_when_missing: bool = True) -> DatasetContext:
    selected = state.get("selected_dataset")
    if isinstance(selected, dict):
        return dataset_context(
            dataset_key=selected.get("dataset_key"),
            dataset_path=selected.get("path"),
            scale_factor=selected.get("scale_factor"),
        )
    if legacy_when_missing:
        return dataset_context(use_legacy=True)
    return dataset_context()


def manifest_matches_selected_dataset(
    manifest: dict[str, Any],
    selected_dataset: dict[str, Any] | None,
) -> bool:
    if not isinstance(selected_dataset, dict):
        return False
    dataset = manifest.get("dataset")
    if not isinstance(dataset, dict):
        return False

    expected_key = str(selected_dataset.get("dataset_key", "")).strip()
    actual_key = str(dataset.get("dataset_key", "")).strip()
    if not expected_key or not actual_key or expected_key != actual_key:
        return False

    expected_path = str(selected_dataset.get("path", "")).strip()
    actual_path = str(dataset.get("dataset_path", "")).strip()
    if expected_path and actual_path and expected_path != actual_path:
        return False
    return True


def split_signature_matches_manifest(
    metadata: dict[str, Any] | None,
    manifest: dict[str, Any] | None,
) -> bool | None:
    metadata_sig = split_signature_from_metadata(metadata if isinstance(metadata, dict) else None)
    manifest_sig = split_signature_from_manifest_payload(
        manifest if isinstance(manifest, dict) else None
    )
    if not metadata_sig or not manifest_sig:
        return None
    return metadata_sig == manifest_sig


def result_split_status(
    payload: dict[str, Any] | None,
    manifest: dict[str, Any] | None,
    *,
    metadata_key: str = "metadata",
) -> dict[str, Any]:
    if not isinstance(payload, dict):
        return {
            "has_signature": False,
            "split_match": None,
            "stale": False,
        }
    meta = payload.get(metadata_key)
    if not isinstance(meta, dict):
        meta = payload.get("metadata", {}) if isinstance(payload.get("metadata"), dict) else {}
    result_sig = meta.get("split_signature")
    manifest_sig = split_signature_from_manifest_payload(manifest if isinstance(manifest, dict) else None)
    if not result_sig or not manifest_sig:
        return {
            "has_signature": False,
            "split_match": None,
            "stale": False,
        }
    split_match = str(result_sig) == str(manifest_sig)
    return {
        "has_signature": True,
        "split_match": split_match,
        "stale": not split_match,
    }


def build_file_status(paths: list[Path]) -> dict[str, Any]:
    existing = [rel_path(path) for path in paths if path.exists()]
    missing = [rel_path(path) for path in paths if not path.exists()]
    return {
        "ready": not missing,
        "existing": existing,
        "missing": missing,
        "total": len(paths),
    }


def build_stage_status(
    paths: list[Path],
    *,
    manifest_dir: Path,
    selected_dataset: dict[str, Any] | None,
) -> dict[str, Any]:
    status = build_file_status(paths)
    manifest = load_manifest(manifest_dir)
    manifest_exists = bool(manifest)
    dataset_match = manifest_matches_selected_dataset(manifest, selected_dataset)
    status.update(
        {
            "manifest_exists": manifest_exists,
            "manifest_path": rel_path(manifest_dir / "dataset_manifest.json"),
            "dataset_match": dataset_match,
            "ready": status["ready"] and dataset_match,
        }
    )
    return status


def default_state() -> dict[str, Any]:
    return {
        "version": STATE_VERSION,
        "updated_at": now_iso(),
        "selected_dataset": None,
        "selected_model_target": default_model_target(),
        "selected_benchmark": default_benchmark_target(),
        "workspace": {},
        "last_runs": {},
    }


def ensure_state_shape(state: dict[str, Any]) -> dict[str, Any]:
    base = default_state()
    if not isinstance(state, dict):
        return base
    base.update(state)
    if not isinstance(base.get("last_runs"), dict):
        base["last_runs"] = {}
    if not isinstance(base.get("workspace"), dict):
        base["workspace"] = {}
    selected = base.get("selected_dataset")
    if isinstance(selected, dict):
        selected_path = abs_path(selected.get("path"))
        if selected_path and selected_path.exists():
            selected["path"] = rel_path(selected_path)
            context = dataset_context(
                dataset_key=selected.get("dataset_key"),
                dataset_path=selected_path,
                scale_factor=selected.get("scale_factor"),
            )
            selected["dataset_key"] = context.dataset_key
            selected["scale_factor"] = context.scale_factor
    config, _ = load_tcc_config_safe()
    dataset_key = selected.get("dataset_key") if isinstance(selected, dict) else None
    base["selected_model_target"] = normalize_model_target(
        base.get("selected_model_target"),
        config,
        dataset_key,
    )
    base["selected_benchmark"] = normalize_benchmark_target(
        base.get("selected_benchmark"),
        config,
    )
    return base


def load_state() -> dict[str, Any]:
    if not STATE_PATH.exists():
        return default_state()
    try:
        payload = json.loads(STATE_PATH.read_text(encoding="utf-8"))
    except Exception:
        print("[Aviso] Falha ao ler .pipeline_state.json. Recriando estado padrao.")
        return default_state()
    return ensure_state_shape(payload)


def save_state(state: dict[str, Any]) -> None:
    state["version"] = STATE_VERSION
    state["updated_at"] = now_iso()
    STATE_PATH.write_text(
        json.dumps(state, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def discover_datasets() -> list[dict[str, Any]]:
    datasets: list[dict[str, Any]] = []
    for path in sorted(DATASET_DIR.glob("*.tar.zst")):
        try:
            stat = path.stat()
            datasets.append(
                {
                    "name": path.name,
                    "path": rel_path(path),
                    "dataset_key": dataset_context(dataset_path=path).dataset_key,
                    "scale_factor": detect_scale_factor(path.name),
                    "size_bytes": stat.st_size,
                    "size_human": format_size(stat.st_size),
                    "modified_at": datetime.fromtimestamp(
                        stat.st_mtime,
                        tz=timezone.utc,
                    ).isoformat(),
                }
            )
        except OSError:
            continue
    return datasets


def refresh_state(state: dict[str, Any]) -> dict[str, Any]:
    state = ensure_state_shape(state)
    tcc_config, tcc_error = load_tcc_config_safe()

    datasets = discover_datasets()
    selected = state.get("selected_dataset")
    if isinstance(selected, dict):
        selected_path = abs_path(selected.get("path"))
        exists = bool(selected_path and selected_path.exists())
        selected["exists"] = exists
        if selected_path and selected_path.exists():
            selected_context = dataset_context(
                dataset_key=selected.get("dataset_key"),
                dataset_path=selected_path,
                scale_factor=selected.get("scale_factor"),
            )
            selected["path"] = rel_path(selected_path)
            selected["dataset_key"] = selected_context.dataset_key
            selected["scale_factor"] = selected_context.scale_factor
            try:
                selected["size_bytes"] = selected_path.stat().st_size
            except OSError:
                pass
        else:
            selected_context = dataset_context(
                dataset_key=selected.get("dataset_key"),
                dataset_path=selected.get("path"),
                scale_factor=selected.get("scale_factor"),
            )
        selected["namespace"] = {
            "output_dir": rel_path(selected_context.output_dir),
            "dados_dir": rel_path(selected_context.dados_dir),
            "splits_dir": rel_path(selected_context.splits_dir),
            "models_dir": rel_path(selected_context.models_dir),
            "results_dir": rel_path(selected_context.results_dir),
        }
    else:
        selected_context = current_dataset_context(state)

    selected_dataset_key = (
        selected.get("dataset_key")
        if isinstance(selected, dict)
        else None
    )
    selected_target = normalize_model_target(
        state.get("selected_model_target"),
        tcc_config,
        selected_dataset_key,
    )
    state["selected_model_target"] = selected_target
    selected_model_dir = resolve_model_target_dir(selected_target)
    splits_manifest = load_manifest(selected_context.splits_dir)
    selected_model_status = build_model_dir_status(
        selected_model_dir,
        family=str(selected_target.get("family", "baseline_hibrido")),
        selected_dataset=selected if isinstance(selected, dict) else None,
        split_manifest=splits_manifest,
    )
    selected_family = selected_model_status["family"]
    selected_results_root = target_results_root(selected_target, selected_dataset_key)

    selected_benchmark = normalize_benchmark_target(
        state.get("selected_benchmark"),
        tcc_config,
    )
    state["selected_benchmark"] = selected_benchmark

    extracao_status = build_stage_status(
        [selected_context.output_dir / name for name in EXTRACAO_OUTPUTS],
        manifest_dir=selected_context.output_dir,
        selected_dataset=selected if isinstance(selected, dict) else None,
    )
    dados_status = build_stage_status(
        [selected_context.dados_dir / name for name in TREINAMENTO_DADOS_OUTPUTS],
        manifest_dir=selected_context.dados_dir,
        selected_dataset=selected if isinstance(selected, dict) else None,
    )
    splits_status = build_stage_status(
        [selected_context.splits_dir / name for name in TREINAMENTO_SPLITS_OUTPUTS],
        manifest_dir=selected_context.splits_dir,
        selected_dataset=selected if isinstance(selected, dict) else None,
    )
    modelo_default_status = build_model_dir_status(
        default_model_dir_for_dataset(selected_dataset_key),
        selected_dataset=selected if isinstance(selected, dict) else None,
        split_manifest=splits_manifest,
    )
    offline_payload = load_json_optional(
        selected_results_root / "offline" / "metricas_resumo.json",
        default={},
    )
    offline_result_status = build_file_status(
        [selected_results_root / "offline" / name for name in AVALIACAO_OFFLINE_OUTPUTS]
    )
    offline_result_status.update(result_split_status(offline_payload, splits_manifest))
    if offline_result_status["stale"]:
        offline_result_status["ready"] = False
    popularidade_payload = load_json_optional(
        selected_results_root / "popularidade" / "metricas_antes_depois.json",
        default={},
    )
    popularidade_result_status = build_file_status(
        [selected_results_root / "popularidade" / "metricas_antes_depois.json"]
    )
    popularidade_result_status.update(result_split_status(popularidade_payload, splits_manifest))
    if popularidade_result_status["stale"]:
        popularidade_result_status["ready"] = False
    manual_result_status = build_file_status(
        [selected_results_root / "manual" / "avaliacao_manual.md"]
    )
    otimizacao_result_status = build_file_status(
        [
            selected_results_root / PESOS_EXPERIMENTOS_FILENAME,
            selected_model_dir / PESOS_OTIMOS_FILENAME,
        ]
    )
    benchmark_result_status = build_file_status(
        [selected_context.results_dir / name for name in BENCHMARK_TCC_OUTPUTS]
    )
    benchmark_payload = load_json_optional(
        selected_context.results_dir / "benchmark_modelos.json",
        default={},
    )
    benchmark_manifest_sig = split_signature_from_manifest_payload(splits_manifest)
    benchmark_exec_sig = (
        str(benchmark_payload.get("split_signature_at_execution", "")).strip()
        if isinstance(benchmark_payload, dict)
        else ""
    )
    benchmark_result_status.update(
        {
            "has_signature": bool(benchmark_exec_sig and benchmark_manifest_sig),
            "split_match": (
                benchmark_exec_sig == benchmark_manifest_sig
                if benchmark_exec_sig and benchmark_manifest_sig
                else None
            ),
            "stale": (
                bool(benchmark_exec_sig and benchmark_manifest_sig)
                and benchmark_exec_sig != benchmark_manifest_sig
            ),
        }
    )
    if benchmark_result_status["stale"]:
        benchmark_result_status["ready"] = False
    modelos_experimentos = (
        [path for path in selected_context.models_dir.iterdir() if path.is_dir()]
        if selected_context.models_dir.exists()
        else []
    )
    enabled_tcc_models = tcc_models_from_config(tcc_config, enabled_only=True)

    state["workspace"] = {
        "datasets": datasets,
        "dataset_dir": rel_path(DATASET_DIR),
        "selected_dataset_exists": bool(selected and selected.get("exists")),
        "selected_dataset_context": {
            "output_dir": rel_path(selected_context.output_dir),
            "dados_dir": rel_path(selected_context.dados_dir),
            "splits_dir": rel_path(selected_context.splits_dir),
            "models_dir": rel_path(selected_context.models_dir),
            "results_dir": rel_path(selected_context.results_dir),
        },
        "extracao": extracao_status,
        "treinamento": {
            "dados": dados_status,
            "splits": splits_status,
            "modelo": modelo_default_status["required"],
            "modelo_opcional": modelo_default_status["optional"],
            "alvo": selected_model_status,
            "ready_for_recommendation": selected_model_status["required"]["ready"],
            "ready_for_evaluation": (
                selected_model_status["required"]["ready"] and splits_status["ready"]
            ),
        },
        "avaliacao": {
            "can_run": {
                "offline": selected_model_status["required"]["ready"]
                and (selected_context.splits_dir / "test_interactions.parquet").exists()
                and extracao_status["ready"],
                "popularidade": selected_family == "baseline_hibrido"
                and selected_model_status["required"]["ready"]
                and (selected_context.splits_dir / "val_interactions.parquet").exists(),
                "manual": selected_model_status["required"]["ready"]
                and (AVALIACAO_DIR / "casos_manuais.yaml").exists(),
                "otimizacao": selected_family == "baseline_hibrido"
                and selected_model_status["required"]["ready"]
                and (selected_context.splits_dir / "val_posts.parquet").exists(),
                "benchmark_tcc": TCC_CONFIG_PATH.exists()
                and extracao_status["ready"],
            },
            "resultados": {
                "offline": offline_result_status,
                "popularidade": popularidade_result_status,
                "manual": manual_result_status,
                "otimizacao": otimizacao_result_status,
                "benchmark_tcc": benchmark_result_status,
            },
        },
        "tcc": {
            "config_exists": TCC_CONFIG_PATH.exists(),
            "config_path": rel_path(TCC_CONFIG_PATH),
            "config_error": tcc_error,
            "modelos_dir": rel_path(selected_context.models_dir),
            "num_modelos_experimento": len(modelos_experimentos),
            "num_modelos_habilitados": len(enabled_tcc_models),
            "benchmark_alvo": selected_benchmark,
        },
    }
    state["updated_at"] = now_iso()
    return state


def badge(value: bool) -> str:
    return "OK" if value else "PENDENTE"


def selected_dataset_label(state: dict[str, Any]) -> str:
    selected = state.get("selected_dataset")
    if not isinstance(selected, dict):
        return "nenhum dataset selecionado"
    path = selected.get("path", "(sem caminho)")
    dataset_key = selected.get("dataset_key") or "sem namespace"
    scale_factor = selected.get("scale_factor") or "sem escala detectada"
    source = selected.get("source", "desconhecida")
    exists = selected.get("exists", False)
    suffix = "disponivel" if exists else "ausente"
    return f"{path} [{dataset_key}, {scale_factor}, {source}, {suffix}]"


def get_last_run(state: dict[str, Any], key: str) -> dict[str, Any] | None:
    value = state.get("last_runs", {}).get(key)
    return value if isinstance(value, dict) else None


def extraction_matches_selected_dataset(state: dict[str, Any]) -> bool:
    selected = state.get("selected_dataset")
    extraction = get_last_run(state, "extraction")
    if not isinstance(selected, dict):
        return False
    context = current_dataset_context(state)
    manifest = load_manifest(context.output_dir)
    if manifest and manifest_matches_selected_dataset(manifest, selected):
        return True
    if not extraction:
        return False
    return (
        selected.get("dataset_key") == extraction.get("dataset_key")
        and selected.get("path") == extraction.get("dataset_path")
    )


def print_menu_header(state: dict[str, Any]) -> None:
    workspace = state["workspace"]
    treinamento = workspace["treinamento"]
    avaliacao = workspace["avaliacao"]

    print("\n" + "=" * 72)
    print("Orquestrador do Pipeline Fitness")
    print("=" * 72)
    print(f"Dataset ativo : {selected_dataset_label(state)}")
    print(
        "Modelo alvo  : "
        f"{selected_model_target_label(state)} -> {treinamento['alvo']['model_dir']}"
    )
    print(f"Benchmark TCC : {benchmark_target_label(state)}")
    print(
        "Artefatos     : "
        f"extracao {badge(workspace['extracao']['ready'])} | "
        f"dados {badge(treinamento['dados']['ready'])} | "
        f"splits {badge(treinamento['splits']['ready'])} | "
        f"alvo {badge(treinamento['alvo']['required']['ready'])}"
    )
    print(
        "Avaliacao     : "
        f"offline {badge(avaliacao['can_run']['offline'])} | "
        f"popularidade {badge(avaliacao['can_run']['popularidade'])} | "
        f"manual {badge(avaliacao['can_run']['manual'])} | "
        f"otimizacao {badge(avaliacao['can_run']['otimizacao'])} | "
        f"benchmark TCC {badge(avaliacao['can_run']['benchmark_tcc'])}"
    )
    if not extraction_matches_selected_dataset(state):
        last_extraction = get_last_run(state, "extraction")
        if last_extraction and state.get("selected_dataset"):
            print(
                "Aviso         : o dataset selecionado difere do dataset usado na ultima extracao."
            )
    print("=" * 72)


def print_state_details(state: dict[str, Any]) -> None:
    workspace = state["workspace"]
    treinamento = workspace["treinamento"]
    avaliacao = workspace["avaliacao"]

    print("\n=== Estado atual salvo ===")
    print(f"Arquivo de estado : {rel_path(STATE_PATH)}")
    print(f"Atualizado em     : {state['updated_at']}")
    print(f"Dataset ativo     : {selected_dataset_label(state)}")
    namespace = workspace.get("selected_dataset_context", {})
    if namespace:
        print(f"Namespace output  : {namespace.get('output_dir', '-')}")
        print(f"Namespace dados   : {namespace.get('dados_dir', '-')}")
        print(f"Namespace splits  : {namespace.get('splits_dir', '-')}")
        print(f"Namespace modelos : {namespace.get('models_dir', '-')}")
        print(f"Namespace resultados: {namespace.get('results_dir', '-')}")

    datasets = workspace["datasets"]
    print(f"\nDatasets locais ({len(datasets)} encontrados):")
    for dataset in datasets:
        print(
            f"  - {dataset['path']} "
            f"({dataset.get('scale_factor') or 'sem sf'}, {dataset['size_human']})"
        )
    if not datasets:
        print("  - nenhum dataset encontrado em extracao_filtragem/dataset/")

    print("\nExtracao:")
    print(f"  - Pronta            : {badge(workspace['extracao']['ready'])}")
    print(f"  - Arquivos gerados  : {len(workspace['extracao']['existing'])}/{workspace['extracao']['total']}")

    print("\nTreinamento:")
    print(f"  - Dados preparados  : {badge(treinamento['dados']['ready'])}")
    print(f"  - Splits            : {badge(treinamento['splits']['ready'])}")
    print(f"  - Modelo padrao     : {badge(treinamento['modelo']['ready'])}")
    print(f"  - Pesos otimos      : {badge(treinamento['modelo_opcional']['ready'])}")
    print(f"  - Alvo selecionado  : {selected_model_target_label(state)}")
    print(f"  - Model dir alvo    : {treinamento['alvo']['model_dir']}")
    print(f"  - Família do alvo   : {treinamento['alvo']['family']}")
    print(f"  - Alvo pronto       : {badge(treinamento['alvo']['required']['ready'])}")

    print("\nAvaliacao:")
    for chave, can_run in avaliacao["can_run"].items():
        print(f"  - Pode rodar {chave:<12}: {badge(can_run)}")

    print("\nBenchmark TCC:")
    print(f"  - Configuração JSON : {badge(workspace['tcc']['config_exists'])}")
    print(f"  - Modelos salvos    : {workspace['tcc']['num_modelos_experimento']}")
    print(f"  - Modelos ativos    : {workspace['tcc']['num_modelos_habilitados']}")
    print(f"  - Alvo selecionado  : {benchmark_target_label(state)}")
    if workspace["tcc"].get("config_error"):
        print(f"  - Aviso de config   : {workspace['tcc']['config_error']}")

    print("\nUltimas execucoes registradas:")
    if not state["last_runs"]:
        print("  - nenhuma execucao registrada ainda")
    else:
        for key, payload in state["last_runs"].items():
            at = payload.get("at", "sem horario")
            print(f"  - {key}: {at}")


def choose_option(options: list[str], zero_label: str | None = None) -> int | None:
    for idx, option in enumerate(options, start=1):
        print(f"{idx}. {option}")
    if zero_label is not None:
        print(f"0. {zero_label}")

    while True:
        raw = input("Escolha uma opcao: ").strip()
        if zero_label is not None and raw == "0":
            return None
        if raw.isdigit():
            index = int(raw)
            if 1 <= index <= len(options):
                return index - 1
        print(INVALID_OPTION_MESSAGE)


def choose_multiple_options(options: list[str], zero_label: str | None = None) -> list[int] | None:
    for idx, option in enumerate(options, start=1):
        print(f"{idx}. {option}")
    if zero_label is not None:
        print(f"0. {zero_label}")

    while True:
        raw = input("Escolha uma ou mais opções (ex.: 1,3,4): ").strip()
        if zero_label is not None and raw == "0":
            return None
        parts = [part.strip() for part in raw.split(",") if part.strip()]
        if not parts:
            print(INVALID_OPTION_MESSAGE)
            continue
        if all(part.isdigit() and 1 <= int(part) <= len(options) for part in parts):
            indices: list[int] = []
            for part in parts:
                idx = int(part) - 1
                if idx not in indices:
                    indices.append(idx)
            return indices
        print(INVALID_OPTION_MESSAGE)


def ask_yes_no(prompt: str, default: bool = True) -> bool:
    suffix = "[S/n]" if default else "[s/N]"
    while True:
        raw = input(f"{prompt} {suffix} ").strip().lower()
        if not raw:
            return default
        if raw in {"s", "sim", "y", "yes"}:
            return True
        if raw in {"n", "nao", "não", "no"}:
            return False
        print("Resposta invalida. Digite s ou n.")


def ask_float(prompt: str, default: float) -> float:
    while True:
        raw = input(f"{prompt} [{default}]: ").strip()
        if not raw:
            return default
        try:
            return float(raw.replace(",", "."))
        except ValueError:
            print("Valor invalido. Digite um numero.")


def ask_int(prompt: str, default: int) -> int:
    while True:
        raw = input(f"{prompt} [{default}]: ").strip()
        if not raw:
            return default
        try:
            return int(raw)
        except ValueError:
            print("Valor invalido. Digite um inteiro.")


def run_python_script(script_path: Path, args: list[str] | None = None) -> None:
    args = args or []
    cmd = [sys.executable, str(script_path), *args]
    printable = " ".join([rel_path(script_path), *args])
    print(f"\n[Execucao] python {printable}\n")
    subprocess.run(cmd, cwd=ROOT, check=True)


def register_run(state: dict[str, Any], key: str, payload: dict[str, Any] | None = None) -> None:
    payload = payload or {}
    state["last_runs"][key] = {"at": now_iso(), **payload}


def update_and_save(state: dict[str, Any]) -> dict[str, Any]:
    state = refresh_state(state)
    save_state(state)
    return state


def prompt_model_target_selection(state: dict[str, Any]) -> dict[str, Any]:
    state = update_and_save(state)
    config, error = load_tcc_config_safe()
    dataset_key = (
        state.get("selected_dataset", {}).get("dataset_key")
        if isinstance(state.get("selected_dataset"), dict)
        else None
    )
    default_target = default_model_target(dataset_key)
    options: list[str] = [f"Modelo padrão ({default_target['model_dir']})"]
    payloads: list[dict[str, Any]] = [default_target]

    if error:
        print(f"[Aviso] {error}")
    else:
        for model_cfg in tcc_models_from_config(config, enabled_only=False):
            enabled_suffix = "" if bool(model_cfg.get("enabled", True)) else " [desabilitado]"
            descricao = str(model_cfg.get("descricao", "")).strip()
            descricao_suffix = f" - {descricao}" if descricao else ""
            options.append(
                f"{model_cfg['id']} ({model_cfg.get('family', 'baseline_hibrido')})"
                f"{enabled_suffix}{descricao_suffix}"
            )
            payloads.append(make_experiment_model_target(model_cfg, dataset_key=dataset_key))

    print("\nSelecione o alvo de modelo/experimento:")
    choice = choose_option(options, zero_label="Voltar")
    if choice is None:
        return state

    selected = dict(payloads[choice])
    selected["selected_at"] = now_iso()
    state["selected_model_target"] = normalize_model_target(selected, config, dataset_key)
    state = update_and_save(state)
    print(f"Alvo atualizado para: {selected_model_target_label(state)}")
    return state


def prompt_benchmark_selection(state: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any] | None]:
    state = update_and_save(state)
    config, error = load_tcc_config_safe()
    if error:
        print(f"[Erro] {error}")
        return state, None

    enabled_models = tcc_models_from_config(config, enabled_only=True)
    if not enabled_models:
        print("Nenhum modelo habilitado foi encontrado em casos_uso_tcc.json.")
        return state, None

    print("\nSelecione o escopo do benchmark TCC:")
    scope_choice = choose_option(
        [
            "Rodar todos os modelos habilitados",
            "Escolher um subconjunto de modelos habilitados",
        ],
        zero_label="Voltar",
    )
    if scope_choice is None:
        return state, None

    if scope_choice == 0:
        state["selected_benchmark"] = {
            "scope": "all",
            "model_ids": [],
            "selected_at": now_iso(),
        }
        state = update_and_save(state)
        print("Benchmark configurado para rodar todos os modelos habilitados.")
        return state, state["selected_benchmark"]

    labels = [
        f"{model_cfg['id']} ({model_cfg.get('family', 'baseline_hibrido')}) - "
        f"{str(model_cfg.get('descricao', '')).strip() or 'sem descrição'}"
        for model_cfg in enabled_models
    ]
    selected_indices = choose_multiple_options(labels, zero_label="Voltar")
    if selected_indices is None:
        return state, None

    model_ids = [str(enabled_models[idx]["id"]) for idx in selected_indices]
    state["selected_benchmark"] = {
        "scope": "subset",
        "model_ids": model_ids,
        "selected_at": now_iso(),
    }
    state = update_and_save(state)
    print(f"Benchmark configurado para: {benchmark_target_label(state)}")
    return state, state["selected_benchmark"]


def select_existing_dataset(state: dict[str, Any]) -> dict[str, Any]:
    datasets = state["workspace"]["datasets"]
    if not datasets:
        print("Nenhum dataset encontrado em extracao_filtragem/dataset/.")
        return state

    print("\nSelecione o dataset ja baixado:")
    labels = [
        f"{dataset['name']} ({dataset.get('scale_factor') or 'sem sf'}, {dataset['size_human']})"
        for dataset in datasets
    ]
    choice = choose_option(labels, zero_label="Voltar")
    if choice is None:
        return state

    dataset = datasets[choice]
    update_selected_dataset(
        state,
        abs_path(dataset["path"]) or (DATASET_DIR / dataset["name"]),
        scale_factor=dataset.get("scale_factor"),
        source="existente",
    )
    state = update_and_save(state)
    print(f"Dataset ativo atualizado para: {dataset['path']}")
    return state


def download_dataset_by_scale_factor(
    state: dict[str, Any],
    scale_factor: str,
    *,
    source: str = "download",
) -> dict[str, Any]:
    option = find_download_option(scale_factor)
    if option is None:
        print(f"[Erro] Scale factor não suportado: {scale_factor}")
        return update_and_save(state)

    dataset_path = DATASET_DIR / option["filename"]
    if dataset_path.exists():
        update_selected_dataset(
            state,
            dataset_path,
            scale_factor=option["scale_factor"],
            source="existente",
        )
        register_run(
            state,
            "download",
            {
                "dataset_path": rel_path(dataset_path),
                "dataset_key": dataset_context(dataset_path=dataset_path).dataset_key,
                "scale_factor": option["scale_factor"],
                "source": source,
                "already_present": True,
            },
        )
        print(f"Dataset já disponível localmente: {rel_path(dataset_path)}")
        return update_and_save(state)

    try:
        run_python_script(
            DOWNLOAD_SCRIPT,
            ["--scale-factor", scale_factor],
        )
    except subprocess.CalledProcessError as exc:
        print(f"[Erro] Download interrompido com codigo {exc.returncode}.")
        return update_and_save(state)

    if not dataset_path.exists():
        state = update_and_save(state)
        datasets = state["workspace"]["datasets"]
        for dataset in datasets:
            if dataset.get("name") == option["filename"]:
                dataset_path = abs_path(dataset["path"]) or dataset_path
                break

    if dataset_path.exists():
        update_selected_dataset(
            state,
            dataset_path,
            scale_factor=option["scale_factor"],
            source=source,
        )
        register_run(
            state,
            "download",
            {
                "dataset_path": rel_path(dataset_path),
                "dataset_key": dataset_context(dataset_path=dataset_path).dataset_key,
                "scale_factor": option["scale_factor"],
                "source": source,
            },
        )
        print(f"Dataset baixado/confirmado e selecionado: {rel_path(dataset_path)}")
    else:
        print("[Aviso] O download terminou sem um arquivo detectado automaticamente.")

    return update_and_save(state)


def download_dataset(state: dict[str, Any]) -> dict[str, Any]:
    print("\nOpcoes de download:")
    labels = [
        f"{item['scale_factor']} ({item['size']}) - {item['descricao']}"
        for item in DOWNLOAD_OPTIONS
    ]
    choice = choose_option(labels, zero_label="Voltar")
    if choice is None:
        return state

    option = DOWNLOAD_OPTIONS[choice]
    return download_dataset_by_scale_factor(
        state,
        option["scale_factor"],
        source="download",
    )


def ensure_selected_dataset(state: dict[str, Any]) -> tuple[dict[str, Any], Path | None]:
    state = update_and_save(state)
    selected = state.get("selected_dataset")
    selected_path = abs_path(selected.get("path")) if isinstance(selected, dict) else None
    if selected_path and selected_path.exists():
        return state, selected_path

    selected_scale_factor = selected.get("scale_factor") if isinstance(selected, dict) else None
    if selected_scale_factor and find_download_option(selected_scale_factor):
        print(
            "\nO dataset ativo não está disponível localmente. "
            f"Baixando automaticamente {selected_scale_factor}..."
        )
        state = download_dataset_by_scale_factor(
            state,
            selected_scale_factor,
            source="auto_download",
        )
        selected = state.get("selected_dataset")
        selected_path = abs_path(selected.get("path")) if isinstance(selected, dict) else None
        if selected_path and selected_path.exists():
            return state, selected_path

    print("\nNenhum dataset ativo disponivel no momento.")
    choice = choose_option(
        [
            "Selecionar dataset ja baixado",
            "Baixar dataset agora",
        ],
        zero_label="Cancelar",
    )
    if choice is None:
        return state, None
    if choice == 0:
        state = select_existing_dataset(state)
    else:
        state = download_dataset(state)

    selected = state.get("selected_dataset")
    selected_path = abs_path(selected.get("path")) if isinstance(selected, dict) else None
    if selected_path and selected_path.exists():
        return state, selected_path
    return state, None


def prompt_split_config() -> dict[str, Any]:
    print("\nConfiguracao do split para treinamento:")
    if ask_yes_no("Usar configuracao padrao (70/15/15, seed 42)?", default=True):
        return dict(DEFAULT_SPLIT_CONFIG)

    while True:
        train = ask_float("Proporcao de treino", DEFAULT_SPLIT_CONFIG["train"])
        val = ask_float("Proporcao de validacao", DEFAULT_SPLIT_CONFIG["val"])
        test = ask_float("Proporcao de teste", DEFAULT_SPLIT_CONFIG["test"])
        total = round(train + val + test, 10)
        if all(0 < value < 1 for value in [train, val, test]) and abs(total - 1.0) <= 1e-6:
            break
        print("As proporcoes devem estar entre 0 e 1 e somar 1.0.")

    seed = ask_int("Seed do split", DEFAULT_SPLIT_CONFIG["seed"])
    return {
        "train": train,
        "val": val,
        "test": test,
        "seed": seed,
    }


def get_selected_model_target(state: dict[str, Any]) -> dict[str, Any]:
    target = state.get("selected_model_target")
    if not isinstance(target, dict):
        return default_model_target()
    return target


def get_tcc_model_config(experiment_id: str) -> tuple[dict[str, Any] | None, str | None]:
    config, error = load_tcc_config_safe()
    if error:
        return None, error
    model_cfg = tcc_model_map(config).get(experiment_id)
    if model_cfg is None:
        return None, f"Experimento '{experiment_id}' não encontrado em casos_uso_tcc.json."
    return model_cfg, None


def build_training_plan(
    target: dict[str, Any],
    split_config: dict[str, Any] | None,
) -> tuple[dict[str, Any] | None, str | None]:
    target_type = str(target.get("type", "modelo_padrao"))
    target_model_dir = resolve_model_target_dir(target)
    if target_type != "experimento_tcc":
        return (
            {
                "target": target,
                "experiment_id": None,
                "family": str(target.get("family", "baseline_hibrido")),
                "model_dir": target_model_dir,
                "split_config": dict(split_config or DEFAULT_SPLIT_CONFIG),
                "training_cfg": {},
                "params": {},
                "dataset_ltr": {},
                "top_k": [5, 10, 20],
                "metric_target": "ndcg@10",
                "avaliacoes": {},
                "notes": "",
            },
            None,
        )

    experiment_id = str(target.get("experiment_id", "")).strip()
    model_cfg, error = get_tcc_model_config(experiment_id)
    if error:
        return None, error

    return (
        {
            "target": target,
            "experiment_id": experiment_id,
            "family": str(model_cfg.get("family", "baseline_hibrido")),
            "model_dir": target_model_dir,
            "split_config": dict(model_cfg.get("split_config", DEFAULT_SPLIT_CONFIG)),
            "training_cfg": dict(model_cfg.get("training", {})),
            "params": dict(model_cfg.get("params", {})),
            "dataset_ltr": dict(model_cfg.get("dataset_ltr", {})),
            "top_k": list(model_cfg.get("top_k", [5, 10, 20])),
            "metric_target": str(model_cfg.get("metric_target", "ndcg@10")),
            "avaliacoes": dict(model_cfg.get("avaliacoes", {})),
            "notes": str(model_cfg.get("notes", "")),
        },
        None,
    )


def merge_experiment_metadata(training_plan: dict[str, Any]) -> None:
    if not training_plan.get("experiment_id"):
        return

    merge_model_metadata(
        training_plan["model_dir"],
        {
            "id": training_plan["experiment_id"],
            "family": training_plan["family"],
            "descricao": training_plan["target"].get("descricao", ""),
            "enabled": bool(training_plan["target"].get("enabled", True)),
            "metric_target": training_plan["metric_target"],
            "top_k": training_plan["top_k"],
            "split_config": training_plan["split_config"],
            "avaliacoes": training_plan["avaliacoes"],
            "notes": training_plan["notes"],
            "params": training_plan["params"],
        },
    )


def write_manual_baseline_weights(model_dir: Path, params: dict[str, Any]) -> Path:
    payload = {
        "w_cos": float(params.get("w_cos", 0.40)),
        "w_cooc": float(params.get("w_cooc", 0.25)),
        "w_time": float(params.get("w_time", 0.15)),
        "w_social": float(params.get("w_social", 0.20)),
        "metric_target": "manual_config",
        "metric_target_value": None,
        "top_k": None,
        "otimizacao": {
            "source": "main.py",
            "timestamp_utc": now_iso(),
        },
    }
    path = model_dir / PESOS_OTIMOS_FILENAME
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def target_results_root(target: dict[str, Any], dataset_key: str | None = None) -> Path:
    context = dataset_context(dataset_key=dataset_key)
    if str(target.get("type", "modelo_padrao")) == "experimento_tcc":
        experiment_id = str(target.get("experiment_id", "")).strip()
        if experiment_id:
            return context.results_dir / "modelos" / experiment_id
    if context.is_legacy:
        return RESULTADOS_DIR
    return context.results_dir / "modelo_padrao"


def choose_evaluation_modes(target: dict[str, Any]) -> list[str] | None:
    family = str(target.get("family", "baseline_hibrido"))
    print("\nSelecione o tipo de avaliacao:")
    options = ["Avaliacao offline do recomendador"]
    mode_map: list[list[str]] = [["offline"]]

    if family == "baseline_hibrido":
        options.append("Impacto do peso de popularidade")
        mode_map.append(["popularidade"])

    options.append("Avaliacao manual reproduzivel")
    mode_map.append(["manual"])

    if family == "baseline_hibrido":
        options.append("Otimizacao de pesos")
        mode_map.append(["otimizacao"])

    baseline_all = ["offline", "popularidade", "manual"]
    ltr_all = ["offline", "manual"]
    options.append("Rodar todas as avaliações compatíveis")
    mode_map.append(baseline_all if family == "baseline_hibrido" else ltr_all)

    choice = choose_option(options, zero_label="Voltar")
    if choice is None:
        return None
    return mode_map[choice]


def print_stage_context(state: dict[str, Any], stage_name: str) -> None:
    print(f"\n=== Contexto antes de {stage_name} ===")
    print(f"Dataset ativo        : {selected_dataset_label(state)}")
    print(f"Modelo alvo         : {selected_model_target_label(state)}")
    print(f"Benchmark TCC       : {benchmark_target_label(state)}")
    last_extraction = get_last_run(state, "extraction")
    last_training = get_last_run(state, "training")
    print(
        "Ultima extracao      : "
        f"{last_extraction.get('dataset_path', 'nao registrada') if last_extraction else 'nao registrada'}"
    )
    print(
        "Ultimo treinamento   : "
        f"{last_training.get('at', 'nao registrado') if last_training else 'nao registrado'}"
    )


def run_extraction_sequence(state: dict[str, Any], dataset_path: Path) -> dict[str, Any]:
    print_stage_context(state, "rodar a extracao")
    selected = state.get("selected_dataset") or {}
    try:
        pipeline_args = ["--dataset-path", str(dataset_path)]
        if selected.get("dataset_key"):
            pipeline_args.extend(["--dataset-key", str(selected["dataset_key"])])
        run_python_script(
            PIPELINE_SCRIPT,
            pipeline_args,
        )
    except subprocess.CalledProcessError as exc:
        print(f"[Erro] A extracao falhou com codigo {exc.returncode}.")
        return update_and_save(state)

    register_run(
        state,
        "extraction",
        {
            "dataset_path": rel_path(dataset_path),
            "dataset_key": selected.get("dataset_key"),
            "scale_factor": selected.get("scale_factor"),
            "source": selected.get("source"),
        },
    )
    state = update_and_save(state)
    print("Extracao concluida e estado atualizado.")
    return state


def maybe_align_extraction_with_selected_dataset(state: dict[str, Any]) -> dict[str, Any]:
    selected = state.get("selected_dataset")
    if not isinstance(selected, dict) or not selected.get("exists"):
        return state
    if extraction_matches_selected_dataset(state):
        return state

    if ask_yes_no(
        "O dataset ativo difere do dataset da ultima extracao. Rodar extracao agora para alinhar o contexto?",
        default=True,
    ):
        dataset_path = abs_path(selected.get("path"))
        if dataset_path and dataset_path.exists():
            state = run_extraction_sequence(state, dataset_path)
    return state


def ensure_extraction_ready(state: dict[str, Any]) -> dict[str, Any]:
    state = update_and_save(state)
    if state["workspace"]["extracao"]["ready"]:
        return state

    print("Os artefatos de extracao ainda nao estao prontos.")
    if not ask_yes_no("Deseja rodar a extracao agora?", default=True):
        return state

    state, dataset_path = ensure_selected_dataset(state)
    if dataset_path is None:
        print("Treinamento cancelado porque nao ha dataset ativo disponivel.")
        return state
    return run_extraction_sequence(state, dataset_path)


def run_training_sequence(
    state: dict[str, Any],
    split_config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    state = ensure_extraction_ready(state)
    if not state["workspace"]["extracao"]["ready"]:
        return state

    state = maybe_align_extraction_with_selected_dataset(state)
    state = update_and_save(state)
    if not state["workspace"]["extracao"]["ready"]:
        return state
    if state.get("selected_dataset") and not extraction_matches_selected_dataset(state):
        print("Treinamento cancelado porque a extração não corresponde ao dataset ativo.")
        return state

    target = get_selected_model_target(state)
    training_plan, error = build_training_plan(target, split_config)
    if error:
        print(f"[Erro] {error}")
        return update_and_save(state)
    assert training_plan is not None
    selected = state.get("selected_dataset") or {}
    dataset_key = selected.get("dataset_key") if isinstance(selected, dict) else None
    dataset_path = selected.get("path") if isinstance(selected, dict) else None
    scale_factor = selected.get("scale_factor") if isinstance(selected, dict) else None

    print_stage_context(state, "rodar o treinamento")
    print(f"Split selecionado    : {training_plan['split_config']}")
    print(f"Model dir de saída   : {rel_path(training_plan['model_dir'])}")

    try:
        prepare_args: list[str] = []
        if dataset_key:
            prepare_args.extend(["--dataset-key", str(dataset_key)])
        if dataset_path:
            prepare_args.extend(["--dataset-path", str(dataset_path)])
        if scale_factor:
            prepare_args.extend(["--scale-factor", str(scale_factor)])
        run_python_script(PREPARAR_SCRIPT, prepare_args)
    except subprocess.CalledProcessError as exc:
        print(f"[Erro] preparacao_dados.py falhou com codigo {exc.returncode}.")
        return update_and_save(state)
    register_run(
        state,
        "preparacao_dados",
        {"dataset_key": dataset_key, "dataset_path": dataset_path},
    )
    state = update_and_save(state)

    split_args = [
        "--train",
        str(training_plan["split_config"]["train"]),
        "--val",
        str(training_plan["split_config"]["val"]),
        "--test",
        str(training_plan["split_config"]["test"]),
        "--seed",
        str(training_plan["split_config"]["seed"]),
    ]
    if dataset_key:
        split_args.extend(["--dataset-key", str(dataset_key)])
    if dataset_path:
        split_args.extend(["--dataset-path", str(dataset_path)])
    if scale_factor:
        split_args.extend(["--scale-factor", str(scale_factor)])
    try:
        run_python_script(DIVIDIR_SCRIPT, split_args)
    except subprocess.CalledProcessError as exc:
        print(f"[Erro] dividir_dataset.py falhou com codigo {exc.returncode}.")
        return update_and_save(state)
    register_run(
        state,
        "split",
        {
            **dict(training_plan["split_config"]),
            "dataset_key": dataset_key,
            "dataset_path": dataset_path,
        },
    )
    state = update_and_save(state)

    train_args = ["--model-dir", str(training_plan["model_dir"])]
    if training_plan["experiment_id"]:
        train_args.extend(
            [
                "--experiment-id",
                str(training_plan["experiment_id"]),
            ]
        )

    training_cfg = training_plan["training_cfg"]
    if training_cfg.get("dataset_completo"):
        train_args.append("--dataset-completo")
    elif training_cfg.get("catalogo_completo", bool(training_plan["experiment_id"])):
        train_args.append("--catalogo-completo")
    if dataset_key:
        train_args.extend(["--dataset-key", str(dataset_key)])
    if dataset_path:
        train_args.extend(["--dataset-path", str(dataset_path)])
    if scale_factor:
        train_args.extend(["--scale-factor", str(scale_factor)])

    try:
        run_python_script(TREINAR_SCRIPT, train_args)
    except subprocess.CalledProcessError as exc:
        print(f"[Erro] treinar.py falhou com codigo {exc.returncode}.")
        return update_and_save(state)
    merge_experiment_metadata(training_plan)
    state = update_and_save(state)

    if training_plan["family"] == "baseline_hibrido":
        params = training_plan["params"]
        if params.get("usar_pesos_otimos"):
            otim_args = [
                "--model-dir",
                str(training_plan["model_dir"]),
                "--grid-step",
                str(params.get("grid_step", 0.1)),
                "--random-search",
                str(params.get("random_search", 0)),
                "--top-k",
                str(params.get("otimizacao_top_k", 10)),
                "--max-queries",
                str(params.get("max_queries_otimizacao", 300)),
                "--seed",
                str(training_plan["split_config"]["seed"]),
            ]
            if training_plan["experiment_id"]:
                otim_args.extend(
                    [
                        "--out-csv",
                        str(target_results_root(target, dataset_key) / PESOS_EXPERIMENTOS_FILENAME),
                        "--out-json",
                        str(training_plan["model_dir"] / PESOS_OTIMOS_FILENAME),
                    ]
                )
            if dataset_key:
                otim_args.extend(["--dataset-key", str(dataset_key)])
            try:
                run_python_script(OTIMIZAR_PESOS_SCRIPT, otim_args)
            except subprocess.CalledProcessError as exc:
                print(f"[Erro] otimizar_pesos.py falhou com codigo {exc.returncode}.")
                return update_and_save(state)
        elif training_plan["experiment_id"]:
            write_manual_baseline_weights(training_plan["model_dir"], params)

    if training_plan["family"] == "ltr_lightgbm":
        dataset_cfg = training_plan["dataset_ltr"]
        features_enabled = dataset_cfg.get("features_enabled", [])
        ltr_dataset_args = [
            "--model-dir",
            str(training_plan["model_dir"]),
            "--train-out",
            str(training_plan["model_dir"] / "ltr_train.parquet"),
            "--val-out",
            str(training_plan["model_dir"] / "ltr_val.parquet"),
            "--meta-out",
            str(training_plan["model_dir"] / "ltr_dataset_meta.json"),
            "--negatives-per-query",
            str(dataset_cfg.get("negatives_per_query", 50)),
            "--hard-negative-topn",
            str(dataset_cfg.get("hard_negative_topn", 500)),
            "--max-queries-train",
            str(dataset_cfg.get("max_queries_train", 500)),
            "--max-queries-val",
            str(dataset_cfg.get("max_queries_val", 200)),
            "--seed",
            str(dataset_cfg.get("seed", training_plan["split_config"]["seed"])),
        ]
        if dataset_key:
            ltr_dataset_args.extend(["--dataset-key", str(dataset_key)])
        if isinstance(features_enabled, list) and features_enabled:
            ltr_dataset_args.extend(["--features", *[str(item) for item in features_enabled]])

        try:
            run_python_script(PREPARAR_LTR_SCRIPT, ltr_dataset_args)
        except subprocess.CalledProcessError as exc:
            print(f"[Erro] preparar_dataset_ltr.py falhou com codigo {exc.returncode}.")
            return update_and_save(state)

        params = training_plan["params"]
        ltr_train_args = [
            "--model-dir",
            str(training_plan["model_dir"]),
            "--train-dataset",
            str(training_plan["model_dir"] / "ltr_train.parquet"),
            "--val-dataset",
            str(training_plan["model_dir"] / "ltr_val.parquet"),
            "--meta-dataset",
            str(training_plan["model_dir"] / "ltr_dataset_meta.json"),
            "--objective",
            str(params.get("objective", "lambdarank")),
            "--metric-at",
            *[str(k) for k in params.get("metric_at", training_plan["top_k"])],
            "--num-leaves",
            str(params.get("num_leaves", 31)),
            "--learning-rate",
            str(params.get("learning_rate", 0.05)),
            "--n-estimators",
            str(params.get("n_estimators", 300)),
            "--min-data-in-leaf",
            str(params.get("min_data_in_leaf", 20)),
            "--feature-fraction",
            str(params.get("feature_fraction", 0.9)),
            "--bagging-fraction",
            str(params.get("bagging_fraction", 0.8)),
            "--bagging-freq",
            str(params.get("bagging_freq", 1)),
            "--seed",
            str(params.get("seed", training_plan["split_config"]["seed"])),
        ]
        try:
            run_python_script(TREINAR_LTR_SCRIPT, ltr_train_args)
        except subprocess.CalledProcessError as exc:
            print(f"[Erro] treinar_ltr.py falhou com codigo {exc.returncode}.")
            return update_and_save(state)

    selected = state.get("selected_dataset") or {}
    last_extraction = get_last_run(state, "extraction") or {}
    register_run(
        state,
        "training",
        {
            "split_config": dict(training_plan["split_config"]),
            "selected_dataset_path": selected.get("path"),
            "dataset_key": selected.get("dataset_key"),
            "extraction_dataset_path": last_extraction.get("dataset_path"),
            "model_target": dict(target),
            "model_dir": rel_path(training_plan["model_dir"]),
            "family": training_plan["family"],
        },
    )
    state = update_and_save(state)
    print("Treinamento concluido e estado atualizado.")
    return state


def modes_can_run(state: dict[str, Any], modes: list[str]) -> bool:
    can_run = state["workspace"]["avaliacao"]["can_run"]
    return all(can_run.get(mode, False) for mode in modes)


def run_evaluation_sequence(state: dict[str, Any], modes: list[str]) -> dict[str, Any]:
    state = update_and_save(state)
    target = get_selected_model_target(state)
    family = str(target.get("family", "baseline_hibrido"))
    selected = state.get("selected_dataset") or {}
    dataset_key = selected.get("dataset_key") if isinstance(selected, dict) else None
    unsupported = [
        mode
        for mode in modes
        if mode in {"popularidade", "otimizacao"} and family != "baseline_hibrido"
    ]
    if unsupported:
        print(
            "Os seguintes modos não são suportados para o alvo atual "
            f"({selected_model_target_label(state)}): {', '.join(unsupported)}"
        )
        return state

    can_run = state["workspace"]["avaliacao"]["can_run"]
    unavailable = [mode for mode in modes if not can_run.get(mode, False)]

    if unavailable:
        print(
            "Os seguintes modos de avaliacao ainda nao podem rodar: "
            + ", ".join(unavailable)
        )
        if ask_yes_no(
            "Deseja rodar o treinamento do alvo atual agora para preparar o contexto?",
            default=True,
        ):
            split_config = (
                None
                if str(target.get("type", "modelo_padrao")) == "experimento_tcc"
                else prompt_split_config()
            )
            state = run_training_sequence(state, split_config)
            state = update_and_save(state)
        if not modes_can_run(state, modes):
            print("A avaliacao foi cancelada porque os pre-requisitos continuam incompletos.")
            return state

    print_stage_context(state, "rodar a avaliacao")
    model_dir = resolve_model_target_dir(target)
    results_root = target_results_root(target, dataset_key)

    for mode in modes:
        try:
            if mode == "offline":
                args = [
                    "--model-dir",
                    str(model_dir),
                    "--k",
                    "5",
                    "10",
                    "20",
                ]
                if dataset_key:
                    args.extend(["--dataset-key", str(dataset_key)])
                args.extend(["--out-dir", str(results_root / "offline")])
                run_python_script(AVALIAR_MODELO_SCRIPT, args)
                register_run(
                    state,
                    "evaluation_offline",
                    {"k": [5, 10, 20], "model_dir": rel_path(model_dir)},
                )
            elif mode == "popularidade":
                args = [
                    "--model-dir",
                    str(model_dir),
                    "--k",
                    "10",
                    "--peso-depois",
                    "0.10",
                    "--out-json",
                    str(results_root / "popularidade" / "metricas_antes_depois.json"),
                ]
                if dataset_key:
                    args.extend(["--dataset-key", str(dataset_key)])
                run_python_script(
                    AVALIAR_POPULARIDADE_SCRIPT,
                    args,
                )
                register_run(
                    state,
                    "evaluation_popularidade",
                    {
                        "k": 10,
                        "peso_depois": 0.10,
                        "demo": False,
                        "model_dir": rel_path(model_dir),
                    },
                )
            elif mode == "manual":
                args = [
                    "--model-dir",
                    str(model_dir),
                    "--saida",
                    str(results_root / "manual" / "avaliacao_manual.md"),
                    "--saida-json",
                    str(results_root / "manual" / "avaliacao_manual.json"),
                ]
                run_python_script(AVALIACAO_MANUAL_SCRIPT, args)
                register_run(
                    state,
                    "evaluation_manual",
                    {
                        "casos": "avaliacao/casos_manuais.yaml",
                        "model_dir": rel_path(model_dir),
                    },
                )
            elif mode == "otimizacao":
                args = [
                    "--model-dir",
                    str(model_dir),
                    "--grid-step",
                    "0.1",
                    "--top-k",
                    "10",
                    "--max-queries",
                    "300",
                    "--seed",
                    "42",
                    "--out-csv",
                    str(results_root / PESOS_EXPERIMENTOS_FILENAME),
                    "--out-json",
                    str(model_dir / PESOS_OTIMOS_FILENAME),
                ]
                if dataset_key:
                    args.extend(["--dataset-key", str(dataset_key)])
                run_python_script(
                    OTIMIZAR_PESOS_SCRIPT,
                    args,
                )
                register_run(
                    state,
                    "optimization",
                    {
                        "grid_step": 0.1,
                        "top_k": 10,
                        "max_queries": 300,
                        "seed": 42,
                        "model_dir": rel_path(model_dir),
                    },
                )
        except subprocess.CalledProcessError as exc:
            print(f"[Erro] A etapa de avaliacao '{mode}' falhou com codigo {exc.returncode}.")
            return update_and_save(state)

        state = update_and_save(state)

    register_run(
        state,
        "evaluation",
        {"modes": modes, "model_dir": rel_path(model_dir), "family": family},
    )
    state = update_and_save(state)
    print("Avaliacao concluida e estado atualizado.")
    return state


def action_run_extraction(state: dict[str, Any]) -> dict[str, Any]:
    state, dataset_path = ensure_selected_dataset(state)
    if dataset_path is None:
        print("Extracao cancelada.")
        return state
    return run_extraction_sequence(state, dataset_path)


def action_select_model_target(state: dict[str, Any]) -> dict[str, Any]:
    return prompt_model_target_selection(state)


def action_run_training(state: dict[str, Any]) -> dict[str, Any]:
    state = update_and_save(state)
    target = get_selected_model_target(state)
    split_config = (
        None
        if str(target.get("type", "modelo_padrao")) == "experimento_tcc"
        else prompt_split_config()
    )
    return run_training_sequence(state, split_config)


def action_run_evaluation(state: dict[str, Any]) -> dict[str, Any]:
    state = update_and_save(state)
    target = get_selected_model_target(state)
    modes = choose_evaluation_modes(target)
    if not modes:
        return state
    return run_evaluation_sequence(state, modes)


def action_run_tcc_benchmark(state: dict[str, Any]) -> dict[str, Any]:
    state = update_and_save(state)
    if not TCC_CONFIG_PATH.exists():
        print("Arquivo casos_uso_tcc.json não encontrado na raiz do projeto.")
        return state

    state, _ = ensure_selected_dataset(state)
    state = ensure_extraction_ready(state)
    if not state["workspace"]["extracao"]["ready"]:
        return state
    state = maybe_align_extraction_with_selected_dataset(state)
    state = update_and_save(state)
    if state.get("selected_dataset") and not extraction_matches_selected_dataset(state):
        print("Benchmark cancelado porque a extração não corresponde ao dataset ativo.")
        return state

    print(f"\nAlvo atual do benchmark: {benchmark_target_label(state)}")
    if ask_yes_no("Deseja reconfigurar quais modelos do benchmark serão executados?", default=False):
        state, selection = prompt_benchmark_selection(state)
        if selection is None:
            return state

    print_stage_context(state, "rodar os casos de uso do TCC")
    selected = state.get("selected_dataset") or {}
    dataset_key = selected.get("dataset_key") if isinstance(selected, dict) else None
    dataset_path = selected.get("path") if isinstance(selected, dict) else None
    scale_factor = selected.get("scale_factor") if isinstance(selected, dict) else None
    benchmark_context = current_dataset_context(state)
    try:
        benchmark_args = ["--config", str(TCC_CONFIG_PATH)]
        if dataset_key:
            benchmark_args.extend(["--dataset-key", str(dataset_key)])
        if dataset_path:
            benchmark_args.extend(["--dataset-path", str(dataset_path)])
        if scale_factor:
            benchmark_args.extend(["--scale-factor", str(scale_factor)])
        benchmark_target = state.get("selected_benchmark", default_benchmark_target())
        if (
            isinstance(benchmark_target, dict)
            and benchmark_target.get("scope") == "subset"
            and benchmark_target.get("model_ids")
        ):
            benchmark_args.extend(
                ["--model-ids", *[str(model_id) for model_id in benchmark_target["model_ids"]]]
            )
        run_python_script(BENCHMARK_TCC_SCRIPT, benchmark_args)
    except subprocess.CalledProcessError as exc:
        print(f"[Erro] benchmark_modelos.py falhou com código {exc.returncode}.")
        return update_and_save(state)

    register_run(
        state,
        "benchmark_tcc",
        {
            "config_path": rel_path(TCC_CONFIG_PATH),
            "dataset_key": dataset_key,
            "outputs": [rel_path(benchmark_context.results_dir / name) for name in BENCHMARK_TCC_OUTPUTS],
            "selection": dict(state.get("selected_benchmark", default_benchmark_target())),
        },
    )
    state = update_and_save(state)
    print("Benchmark TCC concluído e estado atualizado.")
    return state


def action_run_full_pipeline(state: dict[str, Any]) -> dict[str, Any]:
    state, dataset_path = ensure_selected_dataset(state)
    if dataset_path is None:
        print("Fluxo cancelado porque nao ha dataset ativo disponivel.")
        return state

    state = run_extraction_sequence(state, dataset_path)
    state = update_and_save(state)
    if not extraction_matches_selected_dataset(state):
        print("Fluxo interrompido porque a extracao nao concluiu com o dataset ativo.")
        return state

    target = get_selected_model_target(state)
    split_config = (
        None
        if str(target.get("type", "modelo_padrao")) == "experimento_tcc"
        else prompt_split_config()
    )
    state = run_training_sequence(state, split_config)
    state = update_and_save(state)
    if not state["workspace"]["treinamento"]["alvo"]["required"]["ready"]:
        print("Fluxo interrompido porque o treinamento nao concluiu com sucesso.")
        return state

    modes = choose_evaluation_modes(target)
    if not modes:
        return state
    return run_evaluation_sequence(state, modes)


def action_run_training_and_evaluation(state: dict[str, Any]) -> dict[str, Any]:
    state = update_and_save(state)
    target = get_selected_model_target(state)
    split_config = (
        None
        if str(target.get("type", "modelo_padrao")) == "experimento_tcc"
        else prompt_split_config()
    )
    state = run_training_sequence(state, split_config)
    state = update_and_save(state)
    if not state["workspace"]["treinamento"]["alvo"]["required"]["ready"]:
        print("Fluxo interrompido porque o treinamento nao concluiu com sucesso.")
        return state

    modes = choose_evaluation_modes(target)
    if not modes:
        return state
    return run_evaluation_sequence(state, modes)


def main() -> None:
    state = update_and_save(load_state())

    while True:
        print_menu_header(state)
        choice = choose_option(
            [
                "Selecionar dataset ja baixado",
                "Baixar dataset",
                "Selecionar modelo/experimento alvo",
                "Rodar extracao",
                "Rodar treinamento",
                "Rodar avaliacao",
                "Casos de uso do TCC",
                "Rodar extracao + treinamento + avaliacao",
                "Rodar treinamento + avaliacao",
                "Visualizar estado atual salvo",
            ],
            zero_label="Sair",
        )

        if choice is None:
            print("Encerrando o orquestrador.")
            return

        try:
            if choice == 0:
                state = select_existing_dataset(state)
            elif choice == 1:
                state = download_dataset(state)
            elif choice == 2:
                state = action_select_model_target(state)
            elif choice == 3:
                state = action_run_extraction(state)
            elif choice == 4:
                state = action_run_training(state)
            elif choice == 5:
                state = action_run_evaluation(state)
            elif choice == 6:
                state = action_run_tcc_benchmark(state)
            elif choice == 7:
                state = action_run_full_pipeline(state)
            elif choice == 8:
                state = action_run_training_and_evaluation(state)
            elif choice == 9:
                print_state_details(state)
        except KeyboardInterrupt:
            print("\nOperacao interrompida pelo usuario.")

        state = update_and_save(state)


if __name__ == "__main__":
    main()
