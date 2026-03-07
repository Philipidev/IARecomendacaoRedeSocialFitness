from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from treinamento.model_utils import (
    infer_model_family,
    load_model_metadata,
    resolve_model_dir,
)
from treinamento.ranker_features import (
    BaseArtifacts,
    build_feature_frame,
    has_user_profile,
    load_base_artifacts,
    normalize_query_tags,
    score_cooccurrence,
    score_cosine,
    score_popularidade,
    score_social,
    score_time_decay,
    score_user_affinity,
)

PESO_COSINE_PADRAO = 0.40
PESO_COOC_PADRAO = 0.25
PESO_TIME_PADRAO = 0.15
PESO_SOCIAL_PADRAO = 0.20
PESO_POPULARIDADE = 0.10

PESO_COSINE_PERSONALIZADO = 0.30
PESO_COOC_PERSONALIZADO = 0.20
PESO_TIME_PERSONALIZADO = 0.15
PESO_SOCIAL_PERSONALIZADO = 0.15
PESO_USER_AFFINITY = 0.20


def _load_json_optional(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        return payload if isinstance(payload, dict) else {}
    except Exception:
        return {}


class BaseRanker:
    family = "base"

    def __init__(self, model_dir: str | Path | None = None) -> None:
        self.model_dir = resolve_model_dir(model_dir)
        self.metadata: dict[str, Any] = {}
        self.artifacts: BaseArtifacts | None = None
        self._feature_schema: dict[str, Any] = {}

    def carregar(self) -> "BaseRanker":
        raise NotImplementedError

    def recommend_df(
        self,
        tags: list[str],
        timestamp: int,
        top_k: int = 10,
        excluir_tags_exatas: bool = True,
        peso_popularidade: float = PESO_POPULARIDADE,
        user_id: int | None = None,
        include_internal: bool = False,
    ) -> pd.DataFrame:
        raise NotImplementedError


class WeightedHybridRanker(BaseRanker):
    family = "baseline_hibrido"

    def __init__(self, model_dir: str | Path | None = None) -> None:
        super().__init__(model_dir)
        self._vectorizer = None
        self._post_matrix: np.ndarray | None = None
        self._cooccurrence_map: dict[str, list[tuple[str, float]]] | None = None
        self._popularidade: np.ndarray | None = None
        self._social_scores: np.ndarray | None = None
        self._posts: pd.DataFrame | None = None
        self._user_tag_profile: pd.DataFrame | None = None
        self.w_cos = PESO_COSINE_PADRAO
        self.w_cooc = PESO_COOC_PADRAO
        self.w_time = PESO_TIME_PADRAO
        self.w_social = PESO_SOCIAL_PADRAO
        self.peso_popularidade_default = PESO_POPULARIDADE

    def _load_weights(self) -> None:
        default = (
            PESO_COSINE_PADRAO,
            PESO_COOC_PADRAO,
            PESO_TIME_PADRAO,
            PESO_SOCIAL_PADRAO,
        )

        metadata_params = self.metadata.get("params", {})
        if isinstance(metadata_params, dict):
            if metadata_params.get("peso_popularidade") is not None:
                try:
                    self.peso_popularidade_default = float(
                        metadata_params["peso_popularidade"]
                    )
                except Exception:
                    self.peso_popularidade_default = PESO_POPULARIDADE
            explicit = (
                metadata_params.get("w_cos"),
                metadata_params.get("w_cooc"),
                metadata_params.get("w_time"),
                metadata_params.get("w_social"),
            )
            if all(value is not None for value in explicit):
                try:
                    weights = tuple(float(value) for value in explicit)
                    if all(w >= 0 for w in weights) and math.isclose(
                        sum(weights), 1.0, rel_tol=0, abs_tol=1e-6
                    ):
                        self.w_cos, self.w_cooc, self.w_time, self.w_social = weights
                        return
                except Exception:
                    pass

        pesos_path = self.model_dir / "pesos_otimos.json"
        payload = _load_json_optional(pesos_path)
        if payload:
            try:
                weights = (
                    float(payload["w_cos"]),
                    float(payload["w_cooc"]),
                    float(payload["w_time"]),
                    float(payload["w_social"]),
                )
                if all(w >= 0 for w in weights) and math.isclose(
                    sum(weights), 1.0, rel_tol=0, abs_tol=1e-6
                ):
                    self.w_cos, self.w_cooc, self.w_time, self.w_social = weights
                    return
            except Exception:
                pass

        self.w_cos, self.w_cooc, self.w_time, self.w_social = default

    def carregar(self) -> "WeightedHybridRanker":
        self.metadata = load_model_metadata(self.model_dir)
        self.artifacts = load_base_artifacts(self.model_dir)
        self._vectorizer = self.artifacts.vectorizer
        self._post_matrix = self.artifacts.post_matrix
        self._cooccurrence_map = self.artifacts.cooccurrence_map
        self._popularidade = self.artifacts.popularidade
        self._social_scores = self.artifacts.social_scores
        self._posts = self.artifacts.posts_cache
        self._user_tag_profile = self.artifacts.user_tag_profile
        self._load_weights()
        return self

    def _score_cosine(self, tags_entrada: list[str]) -> np.ndarray:
        return score_cosine(self._vectorizer, self._post_matrix, tags_entrada)

    def _score_cooccurrence(self, tags_entrada: list[str]) -> np.ndarray:
        return score_cooccurrence(
            self._cooccurrence_map,
            set(self._vectorizer.classes_),
            self._posts["tags_fitness"],
            tags_entrada,
        )

    def _score_time_decay(self, timestamp_entrada: int) -> np.ndarray:
        return score_time_decay(self._posts, timestamp_entrada)

    def _score_social(self) -> np.ndarray:
        return score_social(self._social_scores, len(self._posts))

    def _score_popularidade(self) -> np.ndarray:
        return score_popularidade(self._popularidade, len(self._posts))

    def _score_user_affinity(self, user_id: int | None) -> np.ndarray:
        return score_user_affinity(self._user_tag_profile, self._posts, user_id)

    def _tem_perfil_usuario(self, user_id: int | None) -> bool:
        return has_user_profile(self._user_tag_profile, user_id)

    def score_candidates(
        self,
        tags: list[str],
        timestamp: int,
        peso_popularidade: float = PESO_POPULARIDADE,
        user_id: int | None = None,
    ) -> np.ndarray:
        if peso_popularidade == PESO_POPULARIDADE:
            peso_popularidade = self.peso_popularidade_default
        tags_norm = normalize_query_tags(tags)
        sc = self._score_cosine(tags_norm)
        si = self._score_cooccurrence(tags_norm)
        st = self._score_time_decay(timestamp)
        ss = self._score_social()
        usar_personalizacao = self._tem_perfil_usuario(user_id)

        if usar_personalizacao:
            su = self._score_user_affinity(user_id)
            score = (
                PESO_COSINE_PERSONALIZADO * sc
                + PESO_COOC_PERSONALIZADO * si
                + PESO_TIME_PERSONALIZADO * st
                + PESO_SOCIAL_PERSONALIZADO * ss
                + PESO_USER_AFFINITY * su
            )
        else:
            sp = self._score_popularidade()
            peso_total = (
                self.w_cos
                + self.w_cooc
                + self.w_time
                + self.w_social
                + peso_popularidade
            )
            score = (
                self.w_cos * sc
                + self.w_cooc * si
                + self.w_time * st
                + self.w_social * ss
                + peso_popularidade * sp
            ) / peso_total

        return np.clip(score, 0.0, 1.0).astype(np.float32)

    def candidate_features(
        self,
        tags: list[str],
        timestamp: int,
        user_id: int | None = None,
    ) -> pd.DataFrame:
        if self.artifacts is None:
            raise RuntimeError("Ranker não carregado.")
        return build_feature_frame(
            self.artifacts,
            tags_entrada=tags,
            timestamp_entrada=timestamp,
            user_id=user_id,
        )

    def recommend_df(
        self,
        tags: list[str],
        timestamp: int,
        top_k: int = 10,
        excluir_tags_exatas: bool = True,
        peso_popularidade: float = PESO_POPULARIDADE,
        user_id: int | None = None,
        include_internal: bool = False,
    ) -> pd.DataFrame:
        tags_norm = normalize_query_tags(tags)
        if not tags_norm:
            raise ValueError("Lista de tags não pode ser vazia.")
        if peso_popularidade == PESO_POPULARIDADE:
            peso_popularidade = self.peso_popularidade_default
        if peso_popularidade < 0:
            raise ValueError("peso_popularidade deve ser maior ou igual a zero.")

        score_final = self.score_candidates(
            tags_norm,
            timestamp,
            peso_popularidade=peso_popularidade,
            user_id=user_id,
        )
        resultado = self._posts.copy()
        resultado["relevance_score"] = score_final.round(4)

        if excluir_tags_exatas:
            tags_set = set(tags_norm)
            resultado = resultado[
                resultado["tags_fitness"].apply(lambda value: set(value) != tags_set)
            ]

        resultado = resultado.sort_values(
            "relevance_score", ascending=False
        ).head(top_k)

        if include_internal:
            return resultado.reset_index(drop=False)

        colunas_saida = [
            "message_type",
            "creation_date_iso",
            "tags_fitness",
            "content_length",
            "language",
            "relevance_score",
        ]
        return resultado[[c for c in colunas_saida if c in resultado.columns]].reset_index(
            drop=True
        )


class LightGBMLTRRanker(BaseRanker):
    family = "ltr_lightgbm"

    def __init__(self, model_dir: str | Path | None = None) -> None:
        super().__init__(model_dir)
        self.booster = None
        self.feature_columns: list[str] = []
        self.categorical_maps: dict[str, dict[str, int]] = {}

    def carregar(self) -> "LightGBMLTRRanker":
        try:
            import lightgbm as lgb  # type: ignore
        except ModuleNotFoundError as exc:  # pragma: no cover - depende do ambiente
            raise ModuleNotFoundError(
                "lightgbm não instalado. Execute: pip install -r requirements.txt"
            ) from exc

        self.metadata = load_model_metadata(self.model_dir)
        self.artifacts = load_base_artifacts(self.model_dir)
        schema_path = self.model_dir / "ltr_feature_schema.json"
        self._feature_schema = _load_json_optional(schema_path)
        self.feature_columns = list(self._feature_schema.get("feature_columns", []))
        self.categorical_maps = dict(
            self._feature_schema.get("categorical_maps", {})
        )
        self.booster = lgb.Booster(model_file=str(self.model_dir / "ltr_model.txt"))
        return self

    def score_candidates(
        self,
        tags: list[str],
        timestamp: int,
        user_id: int | None = None,
    ) -> tuple[np.ndarray, pd.DataFrame]:
        if self.artifacts is None or self.booster is None:
            raise RuntimeError("Ranker não carregado.")

        features_df = build_feature_frame(
            self.artifacts,
            tags_entrada=tags,
            timestamp_entrada=timestamp,
            user_id=user_id,
            categorical_maps=self.categorical_maps or None,
        )
        if not self.feature_columns:
            raise RuntimeError("Schema de features do LTR ausente ou inválido.")
        scores = self.booster.predict(features_df[self.feature_columns]).astype(
            np.float32
        )
        return scores, features_df

    def recommend_df(
        self,
        tags: list[str],
        timestamp: int,
        top_k: int = 10,
        excluir_tags_exatas: bool = True,
        peso_popularidade: float = PESO_POPULARIDADE,
        user_id: int | None = None,
        include_internal: bool = False,
    ) -> pd.DataFrame:
        del peso_popularidade  # não usado na família LTR

        tags_norm = normalize_query_tags(tags)
        if not tags_norm:
            raise ValueError("Lista de tags não pode ser vazia.")

        scores, features_df = self.score_candidates(
            tags_norm, timestamp, user_id=user_id
        )
        resultado = self.artifacts.posts_cache.copy()
        resultado["relevance_score"] = scores.round(4)
        resultado["_catalog_index"] = features_df["catalog_index"].values

        if excluir_tags_exatas:
            tags_set = set(tags_norm)
            resultado = resultado[
                resultado["tags_fitness"].apply(lambda value: set(value) != tags_set)
            ]

        resultado = resultado.sort_values(
            "relevance_score", ascending=False
        ).head(top_k)

        if include_internal:
            return resultado.reset_index(drop=False)

        colunas_saida = [
            "message_type",
            "creation_date_iso",
            "tags_fitness",
            "content_length",
            "language",
            "relevance_score",
        ]
        return resultado[[c for c in colunas_saida if c in resultado.columns]].reset_index(
            drop=True
        )


def load_ranker(model_dir: str | Path | None = None) -> BaseRanker:
    family = infer_model_family(model_dir)
    if family == "ltr_lightgbm":
        return LightGBMLTRRanker(model_dir).carregar()
    return WeightedHybridRanker(model_dir).carregar()
