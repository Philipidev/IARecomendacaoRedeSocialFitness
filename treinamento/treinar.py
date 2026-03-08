"""
Treinamento do modelo de recomendação de posts fitness.

Etapas:
  1. Carrega o catálogo de posts a ser rankeado
  2. Ajusta MultiLabelBinarizer nas tags do conjunto de ajuste
  3. Computa a matriz de posts (post_matrix.npy)
  4. Constrói o mapa de co-ocorrência de tags com pesos normalizados
  5. Calcula popularidade e scores sociais alinhados ao catálogo
  6. Serializa todos os artefatos em um diretório de modelo

Uso:
    # Comportamento legado: catálogo = split de treino
    python treinamento/treinar.py

    # Catálogo completo com estatísticas do split de treino (ideal para benchmark)
    python treinamento/treinar.py --catalogo-completo

    # Dataset completo (catálogo + estatísticas completas)
    python treinamento/treinar.py --dataset-completo
"""

from __future__ import annotations

import argparse
import pickle
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from progress_utils import StageProgress
from dataset_context import DatasetContext, dataset_context, manifest_path
from pipeline_contracts import split_signature_from_manifest_file
from treinamento.model_utils import (
    DEFAULT_MODEL_DIR,
    merge_model_metadata,
    rel_path,
    resolve_model_dir,
)
from treinamento.ranker_features import parse_tags

TRAIN_POSTS_FILE = "train_posts.parquet"


def _parse_tags(value: Any) -> list[str]:
    return parse_tags(value)


def _require_parquet(path: Path, hint: str) -> pd.DataFrame:
    if not path.exists():
        print(f"{path.name} não encontrado. Execute primeiro:")
        print(f"  {hint}")
        sys.exit(1)
    return pd.read_parquet(path)


def _anexar_message_ids(posts: pd.DataFrame, output_dir: Path) -> pd.DataFrame:
    if "_message_id" in posts.columns:
        posts["_message_id"] = pd.to_numeric(
            posts["_message_id"], errors="coerce"
        ).astype("Int64")
        if "message_id" not in posts.columns:
            posts["message_id"] = posts["_message_id"]
        return posts

    if "message_id" in posts.columns:
        posts["message_id"] = pd.to_numeric(
            posts["message_id"], errors="coerce"
        ).astype("Int64")
        posts["_message_id"] = posts["message_id"]
        return posts

    msgs_path = output_dir / "messages_fitness.parquet"
    if not msgs_path.exists():
        return posts

    msgs_df = pd.read_parquet(msgs_path)[["message_id"]].reset_index(drop=True)
    if len(msgs_df) != len(posts):
        return posts

    posts = posts.copy()
    posts["_message_id"] = pd.to_numeric(
        msgs_df["message_id"], errors="coerce"
    ).astype("Int64")
    return posts


def carregar_catalogo_posts(
    usar_split_stats: bool,
    catalogo_completo: bool,
    dados_dir: Path,
    splits_dir: Path,
    output_dir: Path,
) -> tuple[pd.DataFrame, str]:
    if usar_split_stats and not catalogo_completo:
        caminho = splits_dir / TRAIN_POSTS_FILE
        origem = "split de treino"
    elif usar_split_stats and catalogo_completo:
        caminho = dados_dir / "posts_metadata.parquet"
        origem = "catálogo completo com estatísticas de treino"
    else:
        caminho = dados_dir / "posts_metadata.parquet"
        origem = "dataset completo"

    posts = _require_parquet(
        caminho,
        "python treinamento/dividir_dataset.py"
        if caminho.name == TRAIN_POSTS_FILE
        else "python treinamento/preparacao_dados.py",
    )
    posts = posts.copy()
    posts["tags_fitness"] = posts["tags_fitness"].apply(_parse_tags)
    posts = _anexar_message_ids(posts, output_dir)
    return posts, origem


def carregar_posts_ajuste(
    usar_split_stats: bool,
    catalogo_posts: pd.DataFrame,
    catalogo_completo: bool,
    splits_dir: Path,
) -> pd.DataFrame:
    if usar_split_stats and catalogo_completo:
        caminho = splits_dir / TRAIN_POSTS_FILE
        posts_fit = _require_parquet(caminho, "python treinamento/dividir_dataset.py")
        posts_fit = posts_fit.copy()
        posts_fit["tags_fitness"] = posts_fit["tags_fitness"].apply(_parse_tags)
        return posts_fit
    return catalogo_posts


def _filter_known_tags(tags: list[str], vocab: set[str]) -> tuple[list[str], list[str]]:
    known = [tag for tag in tags if tag in vocab]
    unknown = [tag for tag in tags if tag not in vocab]
    return known, unknown


def _catalog_vocabulary_coverage(
    posts_catalogo: pd.DataFrame,
    vocab: set[str],
) -> tuple[list[list[str]], dict[str, Any]]:
    filtered_rows: list[list[str]] = []
    catalog_unique_tags: set[str] = set()
    oov_unique_tags: set[str] = set()
    rows_with_oov = 0
    rows_all_oov = 0
    total_tags = 0
    total_oov_tags = 0

    for tags in posts_catalogo["tags_fitness"]:
        tags_norm = _parse_tags(tags)
        catalog_unique_tags.update(tags_norm)
        total_tags += len(tags_norm)
        known_tags, unknown_tags = _filter_known_tags(tags_norm, vocab)
        if unknown_tags:
            rows_with_oov += 1
            total_oov_tags += len(unknown_tags)
            oov_unique_tags.update(unknown_tags)
        if tags_norm and not known_tags:
            rows_all_oov += 1
        filtered_rows.append(known_tags)

    coverage = {
        "catalog_unique_tags": int(len(catalog_unique_tags)),
        "train_vocabulary_tags": int(len(vocab)),
        "catalog_known_unique_tags": int(len(catalog_unique_tags - oov_unique_tags)),
        "catalog_oov_unique_tags": int(len(oov_unique_tags)),
        "catalog_rows_with_oov": int(rows_with_oov),
        "catalog_rows_all_oov": int(rows_all_oov),
        "catalog_rows_total": int(len(posts_catalogo)),
        "catalog_tags_total": int(total_tags),
        "catalog_oov_tags_total": int(total_oov_tags),
        "catalog_oov_tag_rate": float(total_oov_tags / total_tags) if total_tags > 0 else 0.0,
        "catalog_oov_examples": sorted(oov_unique_tags)[:20],
    }
    return filtered_rows, coverage


def _query_vocabulary_coverage(splits_dir: Path, vocab: set[str]) -> dict[str, Any]:
    query_paths = [
        splits_dir / "val_interactions.parquet",
        splits_dir / "test_interactions.parquet",
    ]
    queries_total = 0
    queries_with_oov = 0
    queries_all_oov = 0
    unique_oov_tags: set[str] = set()

    for path in query_paths:
        if not path.exists():
            continue
        interactions = pd.read_parquet(path)
        if "tags_fitness" not in interactions.columns:
            continue
        for tags in interactions["tags_fitness"].apply(_parse_tags):
            if not tags:
                continue
            queries_total += 1
            known_tags, unknown_tags = _filter_known_tags(tags, vocab)
            if unknown_tags:
                queries_with_oov += 1
                unique_oov_tags.update(unknown_tags)
            if tags and not known_tags:
                queries_all_oov += 1

    return {
        "query_rows_total": int(queries_total),
        "query_rows_with_oov": int(queries_with_oov),
        "query_rows_all_oov": int(queries_all_oov),
        "query_rows_with_oov_rate": float(queries_with_oov / queries_total)
        if queries_total > 0
        else 0.0,
        "query_rows_all_oov_rate": float(queries_all_oov / queries_total)
        if queries_total > 0
        else 0.0,
        "query_oov_examples": sorted(unique_oov_tags)[:20],
    }


def ajustar_vetorizador(
    posts_fit: pd.DataFrame, posts_catalogo: pd.DataFrame
) -> tuple[MultiLabelBinarizer, np.ndarray, set[str], dict[str, Any]]:
    """
    Ajusta MultiLabelBinarizer no conjunto de ajuste e transforma o catálogo.
    """
    mlb = MultiLabelBinarizer()
    mlb.fit(posts_fit["tags_fitness"])
    vocab = {str(tag) for tag in mlb.classes_}
    filtered_tags, coverage = _catalog_vocabulary_coverage(posts_catalogo, vocab)
    post_matrix = mlb.transform(filtered_tags).astype(np.float32)
    print(
        "  Vetorizador: "
        f"{len(mlb.classes_)} tags únicas, "
        f"ajustado em {len(posts_fit)} posts, "
        f"matriz do catálogo {post_matrix.shape}"
    )
    print(
        "  Cobertura do vocabulário: "
        f"{coverage['catalog_rows_with_oov']} posts com OOV, "
        f"{coverage['catalog_rows_all_oov']} posts totalmente fora do vocabulário, "
        f"taxa OOV={coverage['catalog_oov_tag_rate']:.2%}"
    )
    return mlb, post_matrix, vocab, coverage


def construir_cooccurrence_map(
    cooccurrence_df: pd.DataFrame,
) -> dict[str, list[tuple[str, float]]]:
    mapa: dict[str, list[tuple[str, float]]] = defaultdict(list)

    if cooccurrence_df.empty:
        return dict(mapa)

    max_cooc = float(cooccurrence_df["cooccurrences"].max())
    for _, row in cooccurrence_df.iterrows():
        tag_a = str(row["tag_a"])
        tag_b = str(row["tag_b"])
        peso = float(row["cooccurrences"]) / max_cooc if max_cooc > 0 else 0.0
        mapa[tag_a].append((tag_b, peso))
        mapa[tag_b].append((tag_a, peso))

    for tag in mapa:
        mapa[tag] = sorted(mapa[tag], key=lambda item: item[1], reverse=True)

    print(f"  Co-ocorrência: {len(mapa)} tags com vizinhos mapeados")
    return dict(mapa)


def _contagem_tags_interacoes(inter_df: pd.DataFrame) -> dict[str, int]:
    tags_flat = [
        tag
        for tags in inter_df["tags_fitness"].apply(_parse_tags)
        for tag in tags
    ]
    return Counter(tags_flat)


def _scores_por_contagem(posts_catalogo: pd.DataFrame, contagem: dict[str, int]) -> np.ndarray:
    max_inter = max(contagem.values(), default=1)
    scores = [
        sum(contagem.get(tag, 0) for tag in tags) / max_inter
        for tags in posts_catalogo["tags_fitness"]
    ]
    return np.array(scores, dtype=np.float32)


def calcular_popularidade(
    posts_catalogo: pd.DataFrame,
    dados_dir: Path,
    splits_dir: Path,
    usar_split_stats: bool = True,
) -> np.ndarray:
    inter_split_path = splits_dir / "train_interactions.parquet"
    if usar_split_stats and inter_split_path.exists():
        inter_df = pd.read_parquet(inter_split_path)
        contagem = _contagem_tags_interacoes(inter_df)
        if contagem:
            return _scores_por_contagem(posts_catalogo, contagem)
        return np.ones(len(posts_catalogo), dtype=np.float32)

    caminho_pop = dados_dir / "interacoes_por_tag.parquet"
    if not caminho_pop.exists():
        return np.ones(len(posts_catalogo), dtype=np.float32)

    pop_df = pd.read_parquet(caminho_pop)
    pop_map = dict(zip(pop_df["tag_name"], pop_df["total_interacoes"]))
    max_interacoes = max(pop_map.values()) if pop_map else 1

    scores = []
    for tags in posts_catalogo["tags_fitness"]:
        total = sum(pop_map.get(tag, 0) for tag in tags)
        scores.append(total / max_interacoes)

    return np.array(scores, dtype=np.float32)


def _grau_por_usuario(social_graph: pd.DataFrame) -> dict[int, float]:
    degree_as_user = social_graph["user_id"].value_counts()
    degree_as_friend = social_graph["friend_id"].value_counts()
    degree_map = degree_as_user.add(degree_as_friend, fill_value=0)
    return {int(user_id): float(value) for user_id, value in degree_map.items()}


def _calcular_scores_sociais_por_catalogo(
    posts_catalogo: pd.DataFrame,
    interactions: pd.DataFrame,
    social_graph: pd.DataFrame,
) -> np.ndarray:
    if (
        posts_catalogo.empty
        or interactions.empty
        or social_graph.empty
        or "_message_id" not in posts_catalogo.columns
    ):
        return np.zeros(len(posts_catalogo), dtype=np.float32)

    degree_map = _grau_por_usuario(social_graph)
    msg_score = (
        interactions.groupby("message_id")["user_id"]
        .apply(lambda users: float(sum(degree_map.get(int(user_id), 0.0) for user_id in users)))
        .to_dict()
    )

    scores = np.array(
        [
            msg_score.get(int(message_id), 0.0) if pd.notna(message_id) else 0.0
            for message_id in posts_catalogo["_message_id"]
        ],
        dtype=np.float32,
    )
    max_score = float(scores.max()) if len(scores) else 0.0
    if max_score > 0:
        scores /= max_score
    return scores


def carregar_scores_sociais(
    posts_catalogo: pd.DataFrame,
    output_dir: Path,
    dados_dir: Path,
    splits_dir: Path,
    usar_split_stats: bool = True,
) -> np.ndarray:
    if usar_split_stats:
        caminho = splits_dir / "train_social_scores.parquet"
        if caminho.exists():
            ss_df = pd.read_parquet(caminho)
            if len(ss_df) == len(posts_catalogo):
                return (
                    pd.to_numeric(ss_df["social_score"], errors="coerce")
                    .fillna(0.0)
                    .astype(np.float32)
                    .values
                )

        inter_path = splits_dir / "train_interactions.parquet"
        social_path = output_dir / "user_social_graph.parquet"
        if inter_path.exists() and social_path.exists():
            return _calcular_scores_sociais_por_catalogo(
                posts_catalogo,
                pd.read_parquet(inter_path),
                pd.read_parquet(social_path),
            )

        print("  [AVISO] train_social_scores indisponível — social_scores zerados.")
        return np.zeros(len(posts_catalogo), dtype=np.float32)

    caminho = dados_dir / "social_scores.parquet"
    if caminho.exists():
        ss_df = pd.read_parquet(caminho)
        if len(ss_df) == len(posts_catalogo):
            return (
                pd.to_numeric(ss_df["social_score"], errors="coerce")
                .fillna(0.0)
                .astype(np.float32)
                .values
            )

    inter_path = output_dir / "interactions_fitness.parquet"
    social_path = output_dir / "user_social_graph.parquet"
    if inter_path.exists() and social_path.exists():
        return _calcular_scores_sociais_por_catalogo(
            posts_catalogo,
            pd.read_parquet(inter_path),
            pd.read_parquet(social_path),
        )

    print("  [AVISO] social_scores indisponível — social_scores zerados.")
    return np.zeros(len(posts_catalogo), dtype=np.float32)


def carregar_cooccurrence_df(
    output_dir: Path,
    splits_dir: Path,
    usar_split_stats: bool,
) -> pd.DataFrame:
    cooc_split_path = splits_dir / "train_tag_cooccurrence.parquet"
    cooc_full_path = output_dir / "tag_cooccurrence.parquet"

    if usar_split_stats and cooc_split_path.exists():
        cooc_df = pd.read_parquet(cooc_split_path)
        print(f"  Usando co-ocorrência do split de treino ({len(cooc_df)} pares)")
        return cooc_df

    if cooc_full_path.exists():
        cooc_df = pd.read_parquet(cooc_full_path)
        print(f"  Usando co-ocorrência completa ({len(cooc_df)} pares)")
        return cooc_df

    print("  [AVISO] Nenhum arquivo de co-ocorrência encontrado — usando vazio")
    return pd.DataFrame(columns=["tag_a", "tag_b", "cooccurrences"])


def salvar_artefatos(
    model_dir: Path,
    mlb: MultiLabelBinarizer,
    post_matrix: np.ndarray,
    cooccurrence_map: dict[str, Any],
    popularidade: np.ndarray,
    social_scores: np.ndarray,
    posts_catalogo: pd.DataFrame,
) -> None:
    model_dir.mkdir(parents=True, exist_ok=True)

    with open(model_dir / "vectorizer.pkl", "wb") as f:
        pickle.dump(mlb, f)
    print(f"  vectorizer.pkl salvo em {model_dir}")

    np.save(model_dir / "post_matrix.npy", post_matrix)
    print(f"  post_matrix.npy salvo: shape {post_matrix.shape}")

    with open(model_dir / "tag_cooccurrence_map.pkl", "wb") as f:
        pickle.dump(cooccurrence_map, f)
    print("  tag_cooccurrence_map.pkl salvo")

    np.save(model_dir / "popularidade.npy", popularidade)
    print(f"  popularidade.npy salvo: shape {popularidade.shape}")

    np.save(model_dir / "social_scores.npy", social_scores)
    print(f"  social_scores.npy salvo: shape {social_scores.shape}")

    posts_cache = posts_catalogo.copy()
    posts_cache["tags_fitness"] = posts_cache["tags_fitness"].apply(
        lambda tags: tags if isinstance(tags, list) else list(tags)
    )
    posts_cache.to_parquet(model_dir / "posts_cache.parquet", index=True)
    print(f"  posts_cache.parquet salvo: {len(posts_cache)} posts")


def salvar_metadata(
    model_dir: Path,
    args: argparse.Namespace,
    origem: str,
    posts_catalogo: pd.DataFrame,
    posts_fit: pd.DataFrame,
    mlb: MultiLabelBinarizer,
    vocabulary_coverage: dict[str, Any],
    metadata_path: Path | None,
    context: DatasetContext,
) -> None:
    split_manifest = manifest_path(context.splits_dir)
    split_sig = split_signature_from_manifest_file(split_manifest)
    payload: dict[str, Any] = {
        "id": args.experiment_id or model_dir.name,
        "family": "baseline_hibrido",
        "model_dir": rel_path(model_dir),
        "dataset": context.to_metadata(),
        "training": {
            "origem": origem,
            "dataset_completo": bool(args.dataset_completo),
            "catalogo_completo": bool(args.catalogo_completo),
            "usa_estatisticas_split": not bool(args.dataset_completo),
            "n_posts_catalogo": int(len(posts_catalogo)),
            "n_posts_fit": int(len(posts_fit)),
            "n_tags": int(len(mlb.classes_)),
            "split_signature": split_sig,
            "paths": {
                "output_dir": rel_path(context.output_dir),
                "dados_dir": rel_path(context.dados_dir),
                "splits_dir": rel_path(context.splits_dir),
            },
            "vocabulary_coverage": vocabulary_coverage,
        },
        "params": {
            "excluir_tags_exatas": True,
            "peso_popularidade": 0.10,
        },
    }
    merge_model_metadata(model_dir, payload)
    if metadata_path is not None and metadata_path != (model_dir / "metadata.json"):
        metadata_path.parent.mkdir(parents=True, exist_ok=True)
        metadata_path.write_text(
            (model_dir / "metadata.json").read_text(encoding="utf-8"),
            encoding="utf-8",
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Treina o modelo de recomendação de posts fitness.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemplos:
  # Comportamento legado: catálogo = split de treino
  python treinamento/treinar.py

  # Catálogo completo com estatísticas do treino
  python treinamento/treinar.py --catalogo-completo --model-dir treinamento/modelos/baseline_full

  # Dataset completo sem split
  python treinamento/treinar.py --dataset-completo
        """,
    )
    parser.add_argument(
        "--dataset-completo",
        action="store_true",
        default=False,
        help="Usa posts_metadata.parquet e estatísticas completas em vez do split de treino",
    )
    parser.add_argument(
        "--catalogo-completo",
        action="store_true",
        default=False,
        help="Usa posts_metadata.parquet como catálogo, mas mantém estatísticas derivadas do split de treino",
    )
    parser.add_argument(
        "--model-dir",
        type=str,
        default=str(DEFAULT_MODEL_DIR),
        help="Diretório de saída dos artefatos do modelo",
    )
    parser.add_argument(
        "--metadata-path",
        type=str,
        default=None,
        help="Caminho opcional para espelhar o metadata.json do modelo",
    )
    parser.add_argument(
        "--experiment-id",
        type=str,
        default=None,
        help="Identificador lógico do experimento para metadata",
    )
    parser.add_argument(
        "--dataset-key",
        type=str,
        default=None,
        help="Namespace lógico do dataset; se omitido, usa layout legado",
    )
    parser.add_argument(
        "--dataset-path",
        type=str,
        default=None,
        help="Caminho opcional do arquivo do dataset para registrar proveniência",
    )
    parser.add_argument(
        "--scale-factor",
        type=str,
        default=None,
        help="Scale factor opcional para registrar proveniência do dataset",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Override opcional do diretório de extração",
    )
    parser.add_argument(
        "--dados-dir",
        type=str,
        default=None,
        help="Override opcional do diretório de dados preparados",
    )
    parser.add_argument(
        "--splits-dir",
        type=str,
        default=None,
        help="Override opcional do diretório de splits",
    )
    args = parser.parse_args()

    if args.dataset_completo and args.catalogo_completo:
        parser.error("--dataset-completo e --catalogo-completo são mutuamente exclusivos")

    usar_split_stats = not args.dataset_completo
    model_dir = resolve_model_dir(args.model_dir)
    metadata_path = Path(args.metadata_path) if args.metadata_path else None
    base_context = dataset_context(
        dataset_key=args.dataset_key,
        dataset_path=args.dataset_path,
        scale_factor=args.scale_factor,
    )
    output_dir = Path(args.output_dir) if args.output_dir else base_context.output_dir
    if not output_dir.is_absolute():
        output_dir = (ROOT / output_dir).resolve()
    dados_dir = Path(args.dados_dir) if args.dados_dir else base_context.dados_dir
    if not dados_dir.is_absolute():
        dados_dir = (ROOT / dados_dir).resolve()
    splits_dir = Path(args.splits_dir) if args.splits_dir else base_context.splits_dir
    if not splits_dir.is_absolute():
        splits_dir = (ROOT / splits_dir).resolve()
    runtime_context = DatasetContext(
        dataset_key=base_context.dataset_key,
        dataset_path=base_context.dataset_path,
        scale_factor=base_context.scale_factor,
        extraction_dir=base_context.extraction_dir,
        output_dir=output_dir,
        dados_dir=dados_dir,
        splits_dir=splits_dir,
        models_dir=model_dir.parent,
        results_dir=base_context.results_dir,
        is_legacy=base_context.is_legacy,
    )

    print("=== Treinamento do modelo de recomendação ===\n")
    print(f"Namespace ativo : {runtime_context.dataset_key or 'legado'}")
    print(f"Output extração : {runtime_context.output_dir}")
    print(f"Dados treino    : {runtime_context.dados_dir}")
    print(f"Splits          : {runtime_context.splits_dir}")
    print(f"Model dir       : {model_dir}\n")
    progress = StageProgress(
        total_stages=6,
        label=f"Treino {args.experiment_id or model_dir.name}",
    )

    progress.step("Carregando catálogo de posts")
    posts_catalogo, origem = carregar_catalogo_posts(
        usar_split_stats=usar_split_stats,
        catalogo_completo=bool(args.catalogo_completo),
        dados_dir=dados_dir,
        splits_dir=splits_dir,
        output_dir=output_dir,
    )
    print(f"  {len(posts_catalogo)} posts carregados ({origem})")

    posts_fit = carregar_posts_ajuste(
        usar_split_stats=usar_split_stats,
        catalogo_posts=posts_catalogo,
        catalogo_completo=bool(args.catalogo_completo),
        splits_dir=splits_dir,
    )

    print()
    progress.step("Ajustando vetorizador de tags")
    mlb, post_matrix, vocab, vocabulary_coverage = ajustar_vetorizador(posts_fit, posts_catalogo)
    vocabulary_coverage["query_coverage"] = _query_vocabulary_coverage(splits_dir, vocab)

    print()
    progress.step("Carregando co-ocorrência de tags")
    cooc_df = carregar_cooccurrence_df(output_dir, splits_dir, usar_split_stats)
    cooccurrence_map = construir_cooccurrence_map(cooc_df)

    print()
    progress.step("Calculando popularidade")
    popularidade = calcular_popularidade(posts_catalogo, dados_dir, splits_dir, usar_split_stats)
    print(f"  Score médio de popularidade: {popularidade.mean():.4f}")

    print()
    progress.step("Carregando scores de influência social")
    social_scores = carregar_scores_sociais(
        posts_catalogo,
        output_dir,
        dados_dir,
        splits_dir,
        usar_split_stats,
    )
    n_nonzero = int((social_scores > 0).sum())
    print(
        "  Score médio de influência social: "
        f"{social_scores.mean():.4f} ({n_nonzero}/{len(social_scores)} posts com score > 0)"
    )

    print()
    progress.step("Salvando artefatos e metadata")
    salvar_artefatos(
        model_dir=model_dir,
        mlb=mlb,
        post_matrix=post_matrix,
        cooccurrence_map=cooccurrence_map,
        popularidade=popularidade,
        social_scores=social_scores,
        posts_catalogo=posts_catalogo,
    )
    salvar_metadata(
        model_dir=model_dir,
        args=args,
        origem=origem,
        posts_catalogo=posts_catalogo,
        posts_fit=posts_fit,
        mlb=mlb,
        vocabulary_coverage=vocabulary_coverage,
        metadata_path=metadata_path,
        context=runtime_context,
    )

    print(f"\nTreinamento concluído. Artefatos em: {model_dir}")
    print("\nResumo dos artefatos:")
    print(f"  - Fonte dos dados      : {origem}")
    print(f"  - Diretório do modelo  : {rel_path(model_dir)}")
    print(f"  - vectorizer.pkl       : MultiLabelBinarizer com {len(mlb.classes_)} tags")
    print(
        f"  - post_matrix.npy      : {post_matrix.shape[0]} posts × {post_matrix.shape[1]} tags"
    )
    print(f"  - tag_cooccurrence_map : {len(cooccurrence_map)} entradas")
    print(f"  - popularidade.npy     : {len(popularidade)} valores")
    print(f"  - social_scores.npy    : {len(social_scores)} valores")
    print(
        f"  - posts_cache.parquet  : {len(posts_catalogo)} posts (alinhado com post_matrix)"
    )
    print(
        "  - cobertura OOV       : "
        f"{vocabulary_coverage['catalog_rows_with_oov']} posts com OOV, "
        f"{vocabulary_coverage['catalog_rows_all_oov']} totalmente OOV"
    )
    print(
        "  - queries degradadas  : "
        f"{vocabulary_coverage['query_coverage']['query_rows_with_oov']} com OOV, "
        f"{vocabulary_coverage['query_coverage']['query_rows_all_oov']} totalmente OOV"
    )


if __name__ == "__main__":
    main()
