"""
Camada de serviço do simulador FitConnect.

Funciona como um wrapper fino sobre `treinamento/recomendar.py` e
`treinamento/rankers.py`, expondo apenas o que a API HTTP precisa:

- `discover_models()` — varre `treinamento/modelos/<dataset_key>/<model_id>/`
  e retorna metadados resumidos de cada modelo disponível.
- `list_tags(model_dir)` — devolve as tags conhecidas pelo modelo (classes do
  MultiLabelBinarizer) e o tamanho do catálogo de posts.
- `recommend(...)` — chama `recomendar()` reusando o cache global de modelos e
  trata o timestamp default a partir do `posts_cache` carregado.
"""

from __future__ import annotations

import math
from functools import lru_cache
from pathlib import Path
from typing import Any

import pandas as pd

from treinamento.model_utils import load_model_metadata, resolve_model_dir
from treinamento.recomendar import _get_modelo, recomendar

ROOT = Path(__file__).resolve().parent.parent
MODELOS_DIR = ROOT / "treinamento" / "modelos"

_DATASET_ORDER = {"sf0.1": 0, "sf0.3": 1, "sf1": 2, "sf3": 3, "sf10": 4, "sf30": 5}
_FAMILY_ORDER = {
    "popularity": 0,
    "baseline_hibrido": 1,
    "ltr_lightgbm": 2,
}


def _rel(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT.resolve()).as_posix()
    except ValueError:
        return path.as_posix()


def _scale_sort_key(scale_factor: str) -> float:
    if not scale_factor:
        return math.inf
    if scale_factor in _DATASET_ORDER:
        return float(_DATASET_ORDER[scale_factor])
    try:
        return float(scale_factor.replace("sf", ""))
    except Exception:
        return math.inf


def discover_models() -> list[dict[str, Any]]:
    """Lista todos os modelos treinados disponíveis no workspace."""
    if not MODELOS_DIR.exists():
        return []

    encontrados: list[dict[str, Any]] = []
    for dataset_dir in sorted(MODELOS_DIR.iterdir()):
        if not dataset_dir.is_dir():
            continue
        for model_dir in sorted(dataset_dir.iterdir()):
            if not model_dir.is_dir():
                continue
            metadata_path = model_dir / "metadata.json"
            if not metadata_path.exists():
                continue
            metadata = load_model_metadata(model_dir)
            dataset_info = metadata.get("dataset") or {}
            training_info = metadata.get("training") or {}
            family = str(metadata.get("family") or "")
            if not family:
                if (model_dir / "ltr_model.txt").exists():
                    family = "ltr_lightgbm"
                else:
                    family = "baseline_hibrido"

            scale_factor = str(dataset_info.get("scale_factor") or "")
            dataset_key = str(
                dataset_info.get("dataset_key") or dataset_dir.name
            )
            descricao = str(metadata.get("descricao") or "")
            model_id = str(metadata.get("id") or model_dir.name)

            encontrados.append(
                {
                    "model_id": model_id,
                    "model_dir": _rel(model_dir),
                    "dataset_key": dataset_key,
                    "scale_factor": scale_factor,
                    "family": family,
                    "descricao": descricao,
                    "n_posts": int(training_info.get("n_posts_catalogo") or 0),
                    "n_tags": int(training_info.get("n_tags") or 0),
                    "updated_at": metadata.get("updated_at"),
                }
            )

    encontrados.sort(
        key=lambda m: (
            _scale_sort_key(m["scale_factor"]),
            m["dataset_key"],
            _FAMILY_ORDER.get(m["family"], 99),
            m["model_id"],
        )
    )
    return encontrados


def _ranker(model_dir: str | Path):
    """Carrega (com cache) o ranker apropriado para o `model_dir`."""
    return _get_modelo(model_dir)


@lru_cache(maxsize=64)
def _max_timestamp(model_dir_str: str) -> int | None:
    """Retorna o maior `creation_date` do catálogo do modelo."""
    try:
        modelo = _ranker(model_dir_str)
        artifacts = getattr(modelo, "artifacts", None)
        if artifacts is None:
            return None
        posts = artifacts.posts_cache
        if "creation_date" not in posts.columns:
            return None
        valores = pd.to_numeric(posts["creation_date"], errors="coerce").dropna()
        if valores.empty:
            return None
        return int(valores.max())
    except Exception:
        return None


def _vectorizer_classes(model_dir: str | Path) -> list[str]:
    modelo = _ranker(model_dir)
    vectorizer = None
    if hasattr(modelo, "_vectorizer") and getattr(modelo, "_vectorizer") is not None:
        vectorizer = modelo._vectorizer
    elif getattr(modelo, "artifacts", None) is not None:
        vectorizer = modelo.artifacts.vectorizer
    if vectorizer is None:
        return []
    classes = list(getattr(vectorizer, "classes_", []))
    return sorted(str(c) for c in classes)


def list_tags(model_dir: str | Path) -> dict[str, Any]:
    """Lista as tags conhecidas pelo modelo + tamanho do catálogo de posts."""
    resolved = resolve_model_dir(model_dir)
    tags = _vectorizer_classes(resolved)
    modelo = _ranker(resolved)
    artifacts = getattr(modelo, "artifacts", None)
    n_posts = int(len(artifacts.posts_cache)) if artifacts is not None else 0
    max_ts = _max_timestamp(str(resolved.resolve()))
    return {
        "tags": tags,
        "n_tags": len(tags),
        "n_posts": n_posts,
        "default_timestamp": max_ts,
    }


def recommend(
    model_dir: str | Path,
    tags: list[str],
    top_k: int = 20,
    user_id: int | None = None,
    timestamp: int | None = None,
    excluir_tags_exatas: bool = False,
) -> dict[str, Any]:
    """Roda a recomendação no modelo indicado e devolve um payload pronto para JSON."""
    resolved = resolve_model_dir(model_dir)
    if not resolved.exists():
        raise FileNotFoundError(f"Modelo não encontrado: {model_dir}")

    # PopularityRanker ignora a query (tags/timestamp/user) — recomenda os
    # itens mais populares globalmente. Para essa família, tags vazias são
    # válidas. Para baseline_hibrido / ltr_lightgbm, manter a exigência.
    modelo_preview = _ranker(resolved)
    family_preview = str(
        (getattr(modelo_preview, "metadata", {}) or {}).get("family")
        or getattr(modelo_preview, "family", "")
    )
    if not tags and family_preview != "popularity":
        raise ValueError("Selecione ao menos uma tag.")

    ts_usado = timestamp
    if ts_usado is None:
        ts_usado = _max_timestamp(str(resolved.resolve()))
    if ts_usado is None:
        # fallback final: epoch atual em ms — deve ser raro porque todo modelo
        # tem `posts_cache.creation_date`.
        import time

        ts_usado = int(time.time() * 1000)

    df = recomendar(
        tags=list(tags),
        timestamp=int(ts_usado),
        top_k=int(top_k),
        excluir_tags_exatas=bool(excluir_tags_exatas),
        user_id=int(user_id) if user_id is not None else None,
        model_dir=resolved,
    )

    modelo = _ranker(resolved)
    metadata = getattr(modelo, "metadata", {}) or {}
    classes_set = set(_vectorizer_classes(resolved))
    tags_norm = [str(t).strip() for t in tags if str(t).strip()]
    tags_desconhecidas = [t for t in tags_norm if t not in classes_set]

    items: list[dict[str, Any]] = []
    for row in df.to_dict(orient="records"):
        item = {
            "message_type": _safe_str(row.get("message_type")),
            "creation_date_iso": _safe_str(row.get("creation_date_iso")),
            "tags_fitness": [str(t) for t in (row.get("tags_fitness") or []) if t is not None],
            "content_length": _safe_int(row.get("content_length")),
            "language": _safe_str(row.get("language")),
            "relevance_score": _safe_float(row.get("relevance_score")),
        }
        items.append(item)

    artifacts = getattr(modelo, "artifacts", None)
    n_posts = int(len(artifacts.posts_cache)) if artifacts is not None else 0

    return {
        "items": items,
        "meta": {
            "model_id": str(metadata.get("id") or resolved.name),
            "family": str(metadata.get("family") or getattr(modelo, "family", "")),
            "dataset_key": str(
                (metadata.get("dataset") or {}).get("dataset_key") or ""
            ),
            "scale_factor": str(
                (metadata.get("dataset") or {}).get("scale_factor") or ""
            ),
            "timestamp_usado": int(ts_usado),
            "total_catalogo": n_posts,
            "tags_consultadas": tags_norm,
            "tags_desconhecidas": tags_desconhecidas,
            "top_k": int(top_k),
            "personalizado": user_id is not None,
        },
    }


def _safe_int(value: Any) -> int | None:
    try:
        if value is None:
            return None
        if isinstance(value, float) and math.isnan(value):
            return None
        return int(value)
    except Exception:
        return None


def _safe_float(value: Any) -> float | None:
    try:
        if value is None:
            return None
        f = float(value)
        if math.isnan(f) or math.isinf(f):
            return None
        return f
    except Exception:
        return None


def _safe_str(value: Any) -> str | None:
    """Converte para str tratando NaN/None do pandas como None."""
    if value is None:
        return None
    try:
        if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
            return None
    except Exception:
        pass
    try:
        import pandas as _pd

        if _pd.isna(value):
            return None
    except Exception:
        pass
    return str(value)
