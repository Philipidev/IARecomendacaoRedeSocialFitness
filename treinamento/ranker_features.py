from __future__ import annotations

import ast
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

from treinamento.model_utils import DADOS_DIR, load_model_metadata, resolve_model_dir

MS_POR_DIA = 86_400_000
LAMBDA_DECAY = 0.01


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


def normalize_query_tags(tags: list[str]) -> list[str]:
    return [str(tag).strip() for tag in tags if str(tag).strip()]


@dataclass
class BaseArtifacts:
    model_dir: Path
    vectorizer: Any
    post_matrix: np.ndarray
    cooccurrence_map: dict[str, list[tuple[str, float]]]
    popularidade: np.ndarray
    social_scores: np.ndarray
    posts_cache: pd.DataFrame
    user_tag_profile: pd.DataFrame | None
    metadata: dict[str, Any]


def load_base_artifacts(model_dir: str | Path | None = None) -> BaseArtifacts:
    model_path = resolve_model_dir(model_dir)
    metadata = load_model_metadata(model_path)

    with open(model_path / "vectorizer.pkl", "rb") as f:
        vectorizer = pickle.load(f)
    with open(model_path / "tag_cooccurrence_map.pkl", "rb") as f:
        cooccurrence_map = pickle.load(f)

    post_matrix = np.load(model_path / "post_matrix.npy")
    popularidade = np.load(model_path / "popularidade.npy")

    social_path = model_path / "social_scores.npy"
    social_scores = (
        np.load(social_path)
        if social_path.exists()
        else np.zeros(post_matrix.shape[0], dtype=np.float32)
    )

    posts_cache = pd.read_parquet(model_path / "posts_cache.parquet")
    posts_cache = posts_cache.copy()
    if "tags_fitness" in posts_cache.columns:
        posts_cache["tags_fitness"] = posts_cache["tags_fitness"].apply(parse_tags)
    if "_message_id" in posts_cache.columns:
        posts_cache["_message_id"] = pd.to_numeric(
            posts_cache["_message_id"], errors="coerce"
        ).astype("Int64")

    if len(posts_cache) != post_matrix.shape[0]:
        raise ValueError(
            "Inconsistência entre posts_cache.parquet e post_matrix.npy "
            f"({len(posts_cache)} vs {post_matrix.shape[0]})."
        )

    profile_path = DADOS_DIR / "user_tag_profile.parquet"
    user_tag_profile = (
        pd.read_parquet(profile_path) if profile_path.exists() else None
    )

    return BaseArtifacts(
        model_dir=model_path,
        vectorizer=vectorizer,
        post_matrix=post_matrix,
        cooccurrence_map=cooccurrence_map,
        popularidade=popularidade,
        social_scores=social_scores,
        posts_cache=posts_cache,
        user_tag_profile=user_tag_profile,
        metadata=metadata,
    )


def score_cosine(vectorizer: Any, post_matrix: np.ndarray, tags: list[str]) -> np.ndarray:
    query = vectorizer.transform([tags]).astype(np.float32)
    if query.sum() == 0:
        return np.zeros(post_matrix.shape[0], dtype=np.float32)
    return cosine_similarity(query, post_matrix).flatten().astype(np.float32)


def score_cooccurrence(
    cooccurrence_map: dict[str, list[tuple[str, float]]],
    classes_conhecidas: set[str],
    posts_tags: pd.Series,
    tags_entrada: list[str],
) -> np.ndarray:
    boost_por_tag: dict[str, float] = {}
    for tag in tags_entrada:
        for vizinha, peso in cooccurrence_map.get(tag, []):
            if vizinha in classes_conhecidas:
                boost_por_tag[vizinha] = boost_por_tag.get(vizinha, 0.0) + float(peso)

    if not boost_por_tag:
        return np.zeros(len(posts_tags), dtype=np.float32)

    scores = np.zeros(len(posts_tags), dtype=np.float32)
    for i, tags_post in enumerate(posts_tags):
        for tag_post in tags_post:
            scores[i] += boost_por_tag.get(tag_post, 0.0)

    max_score = float(scores.max())
    if max_score > 0:
        scores /= max_score
    return scores


def score_time_decay(posts_cache: pd.DataFrame, timestamp_entrada: int) -> np.ndarray:
    if "creation_date" not in posts_cache.columns:
        return np.ones(len(posts_cache), dtype=np.float32)
    tempos = pd.to_numeric(posts_cache["creation_date"], errors="coerce").fillna(0).values
    delta = np.abs(tempos.astype(np.float64) - float(timestamp_entrada)) / MS_POR_DIA
    return np.exp(-LAMBDA_DECAY * delta).astype(np.float32)


def score_social(social_scores: np.ndarray | None, n_posts: int) -> np.ndarray:
    if social_scores is not None and len(social_scores) == n_posts:
        return social_scores.astype(np.float32)
    return np.zeros(n_posts, dtype=np.float32)


def score_popularidade(popularidade: np.ndarray | None, n_posts: int) -> np.ndarray:
    if popularidade is not None and len(popularidade) == n_posts:
        return popularidade.astype(np.float32)
    return np.zeros(n_posts, dtype=np.float32)


def score_user_affinity(
    user_tag_profile: pd.DataFrame | None,
    posts_cache: pd.DataFrame,
    user_id: int | None,
) -> np.ndarray:
    if user_id is None or user_tag_profile is None or user_tag_profile.empty:
        return np.zeros(len(posts_cache), dtype=np.float32)

    perfil_user = user_tag_profile[user_tag_profile["user_id"] == int(user_id)]
    if perfil_user.empty:
        return np.zeros(len(posts_cache), dtype=np.float32)

    tag_score = dict(
        zip(perfil_user["tag_name"], perfil_user["user_tag_affinity"])
    )
    scores = np.zeros(len(posts_cache), dtype=np.float32)

    for i, tags_post in enumerate(posts_cache["tags_fitness"]):
        if not tags_post:
            continue
        total = sum(float(tag_score.get(tag, 0.0)) for tag in tags_post)
        scores[i] = total / len(tags_post)

    max_score = float(scores.max())
    if max_score > 0:
        scores /= max_score
    return scores


def has_user_profile(user_tag_profile: pd.DataFrame | None, user_id: int | None) -> bool:
    if user_id is None or user_tag_profile is None or user_tag_profile.empty:
        return False
    return bool((user_tag_profile["user_id"] == int(user_id)).any())


def build_categorical_maps(posts_cache: pd.DataFrame) -> dict[str, dict[str, int]]:
    message_values = sorted(
        {
            str(value)
            for value in posts_cache.get(
                "message_type", pd.Series([], dtype="object")
            ).dropna()
        }
    )
    language_values = sorted(
        {
            str(value)
            for value in posts_cache.get(
                "language", pd.Series([], dtype="object")
            ).dropna()
        }
    )
    return {
        "message_type": {value: idx for idx, value in enumerate(message_values, start=1)},
        "language": {value: idx for idx, value in enumerate(language_values, start=1)},
    }


def _safe_float_series(
    values: pd.Series | np.ndarray | list[Any], default: float = 0.0
) -> np.ndarray:
    return pd.to_numeric(pd.Series(values), errors="coerce").fillna(default).astype(
        np.float32
    ).values


def build_feature_frame(
    artifacts: BaseArtifacts,
    tags_entrada: list[str],
    timestamp_entrada: int,
    user_id: int | None = None,
    categorical_maps: dict[str, dict[str, int]] | None = None,
) -> pd.DataFrame:
    posts = artifacts.posts_cache
    tags_norm = normalize_query_tags(tags_entrada)
    sc = score_cosine(artifacts.vectorizer, artifacts.post_matrix, tags_norm)
    si = score_cooccurrence(
        artifacts.cooccurrence_map,
        set(artifacts.vectorizer.classes_),
        posts["tags_fitness"],
        tags_norm,
    )
    st = score_time_decay(posts, timestamp_entrada)
    ss = score_social(artifacts.social_scores, len(posts))
    sp = score_popularidade(artifacts.popularidade, len(posts))
    su = score_user_affinity(artifacts.user_tag_profile, posts, user_id)

    maps = categorical_maps or build_categorical_maps(posts)
    query_set = set(tags_norm)

    tag_overlap_count = []
    tag_jaccard = []
    num_tags_candidate = []
    for tags_post in posts["tags_fitness"]:
        post_set = set(tags_post)
        inter = len(query_set & post_set)
        union = len(query_set | post_set)
        tag_overlap_count.append(inter)
        tag_jaccard.append((inter / union) if union > 0 else 0.0)
        num_tags_candidate.append(len(post_set))

    message_type = posts.get(
        "message_type", pd.Series([None] * len(posts), index=posts.index)
    )
    language = posts.get(
        "language", pd.Series([None] * len(posts), index=posts.index)
    )

    baseline_score = (
        0.40 * sc + 0.25 * si + 0.15 * st + 0.20 * ss + 0.10 * sp
    ) / (0.90 + 0.10)

    return pd.DataFrame(
        {
            "catalog_index": posts.index.astype("int64"),
            "candidate_message_id": pd.to_numeric(
                posts.get("_message_id", pd.Series(posts.index, index=posts.index)),
                errors="coerce",
            ).fillna(-1).astype("int64"),
            "cosine_score": sc,
            "cooccurrence_score": si,
            "time_decay_score": st,
            "social_score": ss,
            "popularidade_score": sp,
            "user_affinity_score": su,
            "tag_overlap_count": np.array(tag_overlap_count, dtype=np.float32),
            "tag_jaccard": np.array(tag_jaccard, dtype=np.float32),
            "num_tags_candidate": np.array(num_tags_candidate, dtype=np.float32),
            "content_length": _safe_float_series(posts.get("content_length", 0.0)),
            "message_type_code": np.array(
                [
                    maps["message_type"].get(str(value), 0)
                    if pd.notna(value)
                    else 0
                    for value in message_type
                ],
                dtype=np.float32,
            ),
            "language_code": np.array(
                [
                    maps["language"].get(str(value), 0)
                    if pd.notna(value)
                    else 0
                    for value in language
                ],
                dtype=np.float32,
            ),
            "baseline_score": baseline_score.astype(np.float32),
        }
    )
