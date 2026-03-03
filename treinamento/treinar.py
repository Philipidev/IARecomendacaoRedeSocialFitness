"""
Treinamento do modelo de recomendação de posts fitness.

Etapas:
  1. Carrega os posts — por padrão usa o split de treino (splits/train_posts.parquet)
  2. Ajusta MultiLabelBinarizer nos nomes das tags
  3. Computa a matriz de posts (post_matrix.npy)
  4. Constrói o mapa de co-ocorrência de tags com pesos normalizados
  5. Carrega os scores de influência social (social_scores.npy)
  6. Serializa todos os artefatos em treinamento/modelo/

Uso:
    # Usa somente o conjunto de treino (recomendado)
    python treinamento/treinar.py

    # Usa o dataset completo (sem divisão)
    python treinamento/treinar.py --dataset-completo
"""

from __future__ import annotations

import argparse
import ast
import os
import pickle
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer

ROOT = Path(__file__).resolve().parent.parent
DADOS_DIR = ROOT / "treinamento" / "dados"
SPLITS_DIR = DADOS_DIR / "splits"
MODELO_DIR = ROOT / "treinamento" / "modelo"
OUTPUT_DIR = ROOT / "extracao_filtragem" / "output"


def _parse_tags(value) -> list[str]:
    import numpy as np
    if isinstance(value, (list, np.ndarray)):
        return [str(t) for t in value]
    if isinstance(value, str):
        try:
            parsed = ast.literal_eval(value)
            return [str(t) for t in parsed] if isinstance(parsed, list) else [value]
        except Exception:
            return [value]
    return []


def carregar_posts(usar_split: bool = True) -> tuple[pd.DataFrame, str]:
    """
    Carrega os posts para treinamento.

    Se usar_split=True (padrão), carrega apenas o conjunto de treino
    (splits/train_posts.parquet). Isso evita que o modelo seja avaliado
    e ajustado com dados que deveriam ser usados só na validação e teste.

    Se usar_split=False, carrega o dataset completo (posts_metadata.parquet),
    útil para treinar o modelo final antes de colocar em produção.
    """
    if usar_split:
        caminho = SPLITS_DIR / "train_posts.parquet"
        if not caminho.exists():
            print("train_posts.parquet não encontrado. Execute primeiro:")
            print("  python treinamento/dividir_dataset.py")
            sys.exit(1)
        origem = "split de treino"
    else:
        caminho = DADOS_DIR / "posts_metadata.parquet"
        if not caminho.exists():
            print("posts_metadata.parquet não encontrado. Execute primeiro:")
            print("  python treinamento/preparacao_dados.py")
            sys.exit(1)
        origem = "dataset completo"

    df = pd.read_parquet(caminho)
    df["tags_fitness"] = df["tags_fitness"].apply(_parse_tags)
    return df, origem


def ajustar_vetorizador(posts: pd.DataFrame) -> tuple[MultiLabelBinarizer, np.ndarray]:
    """
    Ajusta MultiLabelBinarizer e retorna (mlb, post_matrix).
    post_matrix: shape (n_posts, n_tags) — cada linha é o vetor binário do post.
    """
    mlb = MultiLabelBinarizer()
    post_matrix = mlb.fit_transform(posts["tags_fitness"]).astype(np.float32)
    print(f"  Vetorizador: {len(mlb.classes_)} tags únicas, matriz {post_matrix.shape}")
    return mlb, post_matrix


def construir_cooccurrence_map(cooccurrence_df: pd.DataFrame) -> dict[str, list[tuple[str, float]]]:
    """
    Constrói dict: tag_name -> [(tag_relacionada, peso_normalizado), ...]
    O peso é normalizado pelo máximo de co-ocorrências no dataset.
    """
    mapa: dict[str, list[tuple[str, float]]] = defaultdict(list)

    if cooccurrence_df.empty:
        return dict(mapa)

    max_cooc = cooccurrence_df["cooccurrences"].max()

    for _, row in cooccurrence_df.iterrows():
        tag_a = str(row["tag_a"])
        tag_b = str(row["tag_b"])
        peso = float(row["cooccurrences"]) / max_cooc

        mapa[tag_a].append((tag_b, peso))
        mapa[tag_b].append((tag_a, peso))  # relação é simétrica

    # Ordena por peso decrescente
    for tag in mapa:
        mapa[tag] = sorted(mapa[tag], key=lambda x: x[1], reverse=True)

    print(f"  Co-ocorrência: {len(mapa)} tags com vizinhos mapeados")
    return dict(mapa)


def _contagem_tags_interacoes(inter_df: pd.DataFrame) -> dict:
    """Extrai contagem de tags a partir de um DataFrame de interações."""
    from collections import Counter
    tags_flat = [
        tag
        for tags in inter_df["tags_fitness"].apply(_parse_tags)
        for tag in tags
    ]
    return Counter(tags_flat)


def _scores_por_contagem(posts: pd.DataFrame, contagem: dict) -> np.ndarray:
    """Calcula score de popularidade normalizado pelo máximo da contagem."""
    max_inter = max(contagem.values(), default=1)
    scores = [
        sum(contagem.get(t, 0) for t in tags) / max_inter
        for tags in posts["tags_fitness"]
    ]
    return np.array(scores, dtype=np.float32)


def calcular_popularidade(posts: pd.DataFrame, usar_split: bool = True) -> np.ndarray:
    """
    Retorna vetor (n_posts,) com score de popularidade baseado no número de
    interações registradas.
    Quando usar_split=True, usa as interações do split de treino para evitar leakage.
    Posts com tags mais populares recebem score mais alto.
    """
    inter_split_path = SPLITS_DIR / "train_interactions.parquet"
    if usar_split and inter_split_path.exists():
        inter_df = pd.read_parquet(inter_split_path)
        contagem = _contagem_tags_interacoes(inter_df)
        if contagem:
            return _scores_por_contagem(posts, contagem)
        return np.ones(len(posts), dtype=np.float32)

    caminho_pop = DADOS_DIR / "interacoes_por_tag.parquet"
    if not caminho_pop.exists():
        return np.ones(len(posts), dtype=np.float32)

    pop_df = pd.read_parquet(caminho_pop)
    pop_map = dict(zip(pop_df["tag_name"], pop_df["total_interacoes"]))
    max_interacoes = max(pop_map.values()) if pop_map else 1

    scores = []
    for tags in posts["tags_fitness"]:
        total = sum(pop_map.get(t, 0) for t in tags)
        scores.append(total / max_interacoes)

    return np.array(scores, dtype=np.float32)


def carregar_scores_sociais(posts: pd.DataFrame, usar_split: bool = True) -> np.ndarray:
    """
    Carrega o Social Influence Score alinhado com os posts de treinamento.

    Se usar_split=True, usa splits/train_social_scores.parquet para evitar leakage.
    Caso contrário, usa dados/social_scores.parquet (dataset completo).
    Posts sem score recebem 0.

    Retorna vetor float32 de shape (n_posts,).
    """
    if usar_split:
        caminho = SPLITS_DIR / "train_social_scores.parquet"
    else:
        caminho = DADOS_DIR / "social_scores.parquet"

    if not caminho.exists():
        print(f"  [AVISO] {caminho.name} não encontrado — social_scores zerados.")
        return np.zeros(len(posts), dtype=np.float32)

    ss_df = pd.read_parquet(caminho)

    # Alinha pelo índice do posts (0-based após reset em dividir_dataset / preparacao)
    posts_idx = posts.index.tolist()
    scores = np.array(
        [ss_df["social_score"].get(idx, 0.0) if idx in ss_df.index else 0.0 for idx in posts_idx],
        dtype=np.float32,
    )
    return scores


def salvar_artefatos(
    mlb: MultiLabelBinarizer,
    post_matrix: np.ndarray,
    cooccurrence_map: dict,
    popularidade: np.ndarray,
    social_scores: np.ndarray,
    posts: pd.DataFrame,
) -> None:
    MODELO_DIR.mkdir(parents=True, exist_ok=True)

    with open(MODELO_DIR / "vectorizer.pkl", "wb") as f:
        pickle.dump(mlb, f)
    print(f"  vectorizer.pkl salvo em {MODELO_DIR}")

    np.save(MODELO_DIR / "post_matrix.npy", post_matrix)
    print(f"  post_matrix.npy salvo: shape {post_matrix.shape}")

    with open(MODELO_DIR / "tag_cooccurrence_map.pkl", "wb") as f:
        pickle.dump(cooccurrence_map, f)
    print(f"  tag_cooccurrence_map.pkl salvo")

    np.save(MODELO_DIR / "popularidade.npy", popularidade)
    print(f"  popularidade.npy salvo: shape {popularidade.shape}")

    np.save(MODELO_DIR / "social_scores.npy", social_scores)
    print(f"  social_scores.npy salvo: shape {social_scores.shape}")

    # Salva os posts exatos usados no treino para garantir alinhamento em inferência
    posts_cache = posts.drop(columns=["tags_fitness"], errors="ignore").copy()
    posts_cache["tags_fitness"] = posts["tags_fitness"].apply(lambda t: t if isinstance(t, list) else list(t))
    posts_cache.to_parquet(MODELO_DIR / "posts_cache.parquet", index=True)
    print(f"  posts_cache.parquet salvo: {len(posts_cache)} posts")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Treina o modelo de recomendação de posts fitness.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemplos:
  # Usa apenas o conjunto de treino (recomendado)
  python treinamento/treinar.py

  # Usa o dataset completo (para treino final / produção)
  python treinamento/treinar.py --dataset-completo
        """,
    )
    parser.add_argument(
        "--dataset-completo",
        action="store_true",
        default=False,
        help="Usa posts_metadata.parquet completo em vez do split de treino",
    )
    args = parser.parse_args()

    usar_split = not args.dataset_completo

    print("=== Treinamento do modelo de recomendação ===\n")

    print("Carregando posts...")
    posts, origem = carregar_posts(usar_split)
    print(f"  {len(posts)} posts carregados ({origem})")

    print("\nAjustando vetorizador de tags...")
    mlb, post_matrix = ajustar_vetorizador(posts)

    print("\nCarregando co-ocorrência de tags...")
    # Usa a co-ocorrência recalculada só com treino se disponível (evita leakage)
    cooc_split_path = SPLITS_DIR / "train_tag_cooccurrence.parquet"
    cooc_full_path = OUTPUT_DIR / "tag_cooccurrence.parquet"

    if usar_split and cooc_split_path.exists():
        cooc_df = pd.read_parquet(cooc_split_path)
        print(f"  Usando co-ocorrência do split de treino ({len(cooc_df)} pares)")
    elif cooc_full_path.exists():
        cooc_df = pd.read_parquet(cooc_full_path)
        print(f"  Usando co-ocorrência completa ({len(cooc_df)} pares)")
    else:
        print("  [AVISO] Nenhum arquivo de co-ocorrência encontrado — usando vazio")
        cooc_df = pd.DataFrame(columns=["tag_a", "tag_b", "cooccurrences"])

    cooccurrence_map = construir_cooccurrence_map(cooc_df)

    print("\nCalculando popularidade...")
    popularidade = calcular_popularidade(posts, usar_split)
    print(f"  Score médio de popularidade: {popularidade.mean():.4f}")

    print("\nCarregando scores de influência social...")
    social_scores = carregar_scores_sociais(posts, usar_split)
    n_nonzero = int((social_scores > 0).sum())
    print(f"  Score médio de influência social: {social_scores.mean():.4f} ({n_nonzero}/{len(social_scores)} posts com score > 0)")

    print("\nSalvando artefatos...")
    salvar_artefatos(mlb, post_matrix, cooccurrence_map, popularidade, social_scores, posts)

    print(f"\nTreinamento concluído. Artefatos em: {MODELO_DIR}")
    print("\nResumo dos artefatos:")
    print(f"  - Fonte dos dados      : {origem}")
    print(f"  - vectorizer.pkl       : MultiLabelBinarizer com {len(mlb.classes_)} tags")
    print(f"  - post_matrix.npy      : {post_matrix.shape[0]} posts × {post_matrix.shape[1]} tags")
    print(f"  - tag_cooccurrence_map : {len(cooccurrence_map)} entradas")
    print(f"  - popularidade.npy     : {len(popularidade)} valores")
    print(f"  - social_scores.npy    : {len(social_scores)} valores")
    print(f"  - posts_cache.parquet  : {len(posts)} posts (alinhado com post_matrix)")


if __name__ == "__main__":
    main()
