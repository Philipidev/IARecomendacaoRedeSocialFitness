"""
Divisão do dataset em conjuntos de treino, validação e teste.

Divide os posts de forma aleatória (com seed reproduzível) usando percentuais
configuráveis. Funciona independente do tamanho do dataset — sf0.1, sf30 ou
qualquer outro.

Etapas:
  1. Carrega posts_metadata.parquet
  2. Embaralha com seed fixa (reproduzível)
  3. Divide em treino / validação / teste pelos percentuais informados
  4. Filtra as interações correspondentes a cada split
  5. Recalcula tag_cooccurrence usando APENAS os posts de treino (evita data leakage)
  6. Recalcula social_scores usando APENAS as interações de treino (evita data leakage)
  7. Salva todos os splits em treinamento/dados/splits/

Uso:
    python treinamento/dividir_dataset.py
    python treinamento/dividir_dataset.py --train 0.8 --val 0.1 --test 0.1
    python treinamento/dividir_dataset.py --seed 123
"""

from __future__ import annotations

import argparse
import ast
import sys
from collections import defaultdict
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DADOS_DIR = ROOT / "treinamento" / "dados"
SPLITS_DIR = DADOS_DIR / "splits"
OUTPUT_DIR = ROOT / "extracao_filtragem" / "output"

PROPORCAO_TREINO_PADRAO = 0.70
PROPORCAO_VAL_PADRAO = 0.15
PROPORCAO_TESTE_PADRAO = 0.15
SEED_PADRAO = 42


def _parse_tags(value) -> list[str]:
    if isinstance(value, (list, np.ndarray)):
        return [str(t) for t in value]
    if isinstance(value, str):
        try:
            parsed = ast.literal_eval(value)
            return [str(t) for t in parsed] if isinstance(parsed, list) else [value]
        except Exception:
            return [value]
    return []


def validar_proporcoes(treino: float, val: float, teste: float) -> None:
    total = round(treino + val + teste, 10)
    if not (0 < treino < 1 and 0 < val < 1 and 0 < teste < 1):
        print("ERRO: cada proporção deve ser um valor entre 0 e 1 (exclusive).")
        sys.exit(1)
    if abs(total - 1.0) > 1e-6:
        print(f"ERRO: as proporções devem somar 1.0 (soma atual: {total:.6f}).")
        sys.exit(1)


def calcular_cortes(total: int, p_treino: float, p_val: float) -> tuple[int, int]:
    """
    Retorna (n_treino, n_val). O restante vai para teste.
    Qualquer arredondamento é absorvido pelo treino para não perder registros.
    """
    n_val = int(total * p_val)
    n_teste = int(total * (1.0 - p_treino - p_val))
    n_treino = total - n_val - n_teste
    return n_treino, n_val


def dividir_posts(
    posts: pd.DataFrame,
    p_treino: float,
    p_val: float,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Embaralha e divide posts em treino, validação e teste."""
    shuffled = posts.sample(frac=1, random_state=seed).reset_index(drop=True)
    total = len(shuffled)
    n_treino, n_val = calcular_cortes(total, p_treino, p_val)

    train = shuffled.iloc[:n_treino].copy()
    val = shuffled.iloc[n_treino : n_treino + n_val].copy()
    test = shuffled.iloc[n_treino + n_val :].copy()
    return train, val, test


def filtrar_interacoes(
    interactions: pd.DataFrame,
    post_idxs: set,
    coluna_idx: str = "post_idx_original",
) -> pd.DataFrame:
    """Filtra interações cujo post_idx_original pertence ao conjunto dado."""
    return interactions[interactions[coluna_idx].isin(post_idxs)].copy()


def recalcular_cooccurrence(posts_treino: pd.DataFrame) -> pd.DataFrame:
    """
    Recalcula co-ocorrência de tags usando APENAS os posts de treino.
    Evita data leakage: o modelo não aprende relações que só existem no teste.
    """
    contagem: dict[tuple[str, str], int] = defaultdict(int)

    for tags in posts_treino["tags_fitness"]:
        tags_lista = sorted(set(tags))
        for i in range(len(tags_lista)):
            for j in range(i + 1, len(tags_lista)):
                par = (tags_lista[i], tags_lista[j])
                contagem[par] += 1

    if not contagem:
        return pd.DataFrame(columns=["tag_a", "tag_b", "cooccurrences"])

    linhas = [
        {"tag_a": a, "tag_b": b, "cooccurrences": c}
        for (a, b), c in sorted(contagem.items(), key=lambda x: -x[1])
    ]
    return pd.DataFrame(linhas)


def recalcular_social_scores(
    train_df: pd.DataFrame,
    train_interactions: pd.DataFrame,
    social_graph: pd.DataFrame,
) -> pd.DataFrame:
    """
    Recalcula o Social Influence Score usando APENAS dados de treino.
    Evita data leakage: o modelo não aprende influências de interações do teste.

    Parâmetros
    ----------
    train_df : DataFrame do split de treino — deve conter coluna _message_id
    train_interactions : interações filtradas ao split de treino
    social_graph : DataFrame com colunas user_id e friend_id

    Retorna
    -------
    DataFrame com coluna social_score, indexado de 0 a len(train_df)-1.
    """
    if social_graph.empty or train_interactions.empty:
        return pd.DataFrame(
            {"social_score": np.zeros(len(train_df), dtype=np.float32)}
        )

    degree_as_user = social_graph["user_id"].value_counts()
    degree_as_friend = social_graph["friend_id"].value_counts()
    degree_map: dict = degree_as_user.add(degree_as_friend, fill_value=0).to_dict()

    msg_score: dict = (
        train_interactions.groupby("message_id")["user_id"]
        .apply(lambda uids: float(sum(degree_map.get(int(u), 0) for u in uids)))
        .to_dict()
    )

    scores = np.array(
        [msg_score.get(int(mid), 0.0) for mid in train_df["_message_id"].values],
        dtype=np.float32,
    )

    max_score = scores.max()
    if max_score > 0:
        scores /= max_score

    return pd.DataFrame({"social_score": scores})


def imprimir_resumo(
    total: int,
    train: pd.DataFrame,
    val: pd.DataFrame,
    test: pd.DataFrame,
    p_treino: float,
    p_val: float,
    seed: int,
) -> None:
    p_teste = round(1.0 - p_treino - p_val, 10)
    print(f"\n{'='*50}")
    print(f"  Total de posts  : {total}")
    print(f"  Seed utilizada  : {seed}")
    print(f"{'='*50}")
    print(f"  {'Split':<12} {'Proporção':>10} {'Posts':>8} {'Percentual real':>16}")
    print(f"  {'-'*46}")
    for nome, df, prop in [("treino", train, p_treino), ("validação", val, p_val), ("teste", test, p_teste)]:
        pct_real = len(df) / total * 100
        print(f"  {nome:<12} {prop:>9.0%} {len(df):>8}   ({pct_real:.1f}%)")
    print(f"{'='*50}\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Divide o dataset em treino, validação e teste por percentual.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemplos:
  python treinamento/dividir_dataset.py
  python treinamento/dividir_dataset.py --train 0.8 --val 0.1 --test 0.1
  python treinamento/dividir_dataset.py --seed 123
        """,
    )
    parser.add_argument("--train", type=float, default=PROPORCAO_TREINO_PADRAO,
                        help=f"Proporção de treino (padrão: {PROPORCAO_TREINO_PADRAO})")
    parser.add_argument("--val", type=float, default=PROPORCAO_VAL_PADRAO,
                        help=f"Proporção de validação (padrão: {PROPORCAO_VAL_PADRAO})")
    parser.add_argument("--test", type=float, default=PROPORCAO_TESTE_PADRAO,
                        help=f"Proporção de teste (padrão: {PROPORCAO_TESTE_PADRAO})")
    parser.add_argument("--seed", type=int, default=SEED_PADRAO,
                        help=f"Seed aleatória (padrão: {SEED_PADRAO})")
    args = parser.parse_args()

    validar_proporcoes(args.train, args.val, args.test)

    print("=== Divisão do dataset ===")
    print(f"  Proporções : treino={args.train:.0%}  validação={args.val:.0%}  teste={args.test:.0%}")
    print(f"  Seed       : {args.seed}\n")

    # --- Carregar posts ---
    posts_path = DADOS_DIR / "posts_metadata.parquet"
    if not posts_path.exists():
        print("ERRO: posts_metadata.parquet não encontrado.")
        print("Execute primeiro: python treinamento/preparacao_dados.py")
        sys.exit(1)

    posts = pd.read_parquet(posts_path)
    posts["tags_fitness"] = posts["tags_fitness"].apply(_parse_tags)
    # Índice posicional para rastrear a posição de cada post após o embaralhamento
    posts["post_idx_original"] = posts.index
    total = len(posts)
    print(f"  {total} posts carregados de {posts_path.name}")

    # --- Carregar message_id original para cruzar com interações ---
    # posts_metadata não expõe IDs na saída; carregamos messages_fitness para
    # obter o message_id de cada posição e mapear para as interações.
    inter_path = OUTPUT_DIR / "interactions_fitness.parquet"
    msgs_path = OUTPUT_DIR / "messages_fitness.parquet"
    social_path = OUTPUT_DIR / "user_social_graph.parquet"
    has_interactions = inter_path.exists() and msgs_path.exists()
    has_social = social_path.exists()

    if has_interactions:
        interactions = pd.read_parquet(inter_path)
        msgs_raw = pd.read_parquet(msgs_path)[["message_id"]].reset_index(drop=True)
        # Cada linha i de posts_metadata corresponde à linha i de messages_fitness
        posts["_message_id"] = msgs_raw["message_id"].values
        print(f"  {len(interactions)} interações carregadas de {inter_path.name}")
    else:
        print("  [AVISO] interactions_fitness.parquet ou messages_fitness.parquet não encontrado — splits de interações ignorados")

    social_graph = pd.DataFrame()
    if has_social:
        social_graph = pd.read_parquet(social_path)
        print(f"  {len(social_graph)} arestas carregadas de {social_path.name}")
    else:
        print("  [AVISO] user_social_graph.parquet não encontrado — train_social_scores não será gerado")

    # --- Dividir posts ---
    print("\nDividindo posts...")
    train, val, test = dividir_posts(posts, args.train, args.val, args.seed)

    # --- Filtrar interações por split ---
    SPLITS_DIR.mkdir(parents=True, exist_ok=True)

    splits_posts = [("train", train), ("val", val), ("test", test)]
    for nome, df in splits_posts:
        caminho = SPLITS_DIR / f"{nome}_posts.parquet"
        # Remove colunas auxiliares internas antes de salvar
        df.drop(columns=["post_idx_original", "_message_id"], errors="ignore").to_parquet(caminho, index=True)
        print(f"  {nome}_posts.parquet salvo: {len(df)} posts")

    if has_interactions:
        for nome, df in splits_posts:
            msg_ids = set(df["_message_id"].tolist())
            df_inter = interactions[interactions["message_id"].isin(msg_ids)].copy()
            caminho = SPLITS_DIR / f"{nome}_interactions.parquet"
            df_inter.to_parquet(caminho, index=False)
            print(f"  {nome}_interactions.parquet salvo: {len(df_inter)} interações")

    # --- Recalcular co-ocorrência SÓ com treino (evita data leakage) ---
    print("\nRecalculando tag_cooccurrence com dados de treino...")
    cooc_treino = recalcular_cooccurrence(train)
    caminho_cooc = SPLITS_DIR / "train_tag_cooccurrence.parquet"
    cooc_treino.to_parquet(caminho_cooc, index=False)
    print(f"  train_tag_cooccurrence.parquet salvo: {len(cooc_treino)} pares de tags")

    # --- Recalcular social scores SÓ com treino (evita data leakage) ---
    if has_interactions and has_social and "_message_id" in train.columns:
        print("\nRecalculando social_scores com dados de treino...")
        train_msg_ids = set(train["_message_id"].tolist())
        train_inter = interactions[interactions["message_id"].isin(train_msg_ids)].copy()
        social_scores_treino = recalcular_social_scores(train, train_inter, social_graph)
        caminho_social_scores = SPLITS_DIR / "train_social_scores.parquet"
        social_scores_treino.to_parquet(caminho_social_scores, index=True)
        score_medio = social_scores_treino["social_score"].mean()
        print(f"  train_social_scores.parquet salvo: {len(social_scores_treino)} posts")
        print(f"  Score médio de influência social: {score_medio:.4f}")
    else:
        print("\n[AVISO] Grafo social ou interações ausentes — train_social_scores.parquet não gerado.")

    # --- Resumo final ---
    imprimir_resumo(total, train, val, test, args.train, args.val, args.seed)
    print(f"Splits salvos em: {SPLITS_DIR}")


if __name__ == "__main__":
    main()
