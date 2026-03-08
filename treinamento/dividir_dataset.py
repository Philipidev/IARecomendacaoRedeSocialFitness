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
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dataset_context import build_stage_manifest, dataset_context, rel_path, write_manifest
from pipeline_contracts import normalize_split_config, split_signature

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


def garantir_message_ids(posts: pd.DataFrame, output_dir: Path) -> pd.DataFrame:
    posts = posts.copy()
    if "_message_id" in posts.columns:
        posts["_message_id"] = pd.to_numeric(posts["_message_id"], errors="coerce").astype("Int64")
        if "message_id" not in posts.columns:
            posts["message_id"] = posts["_message_id"]
        return posts

    if "message_id" in posts.columns:
        posts["message_id"] = pd.to_numeric(posts["message_id"], errors="coerce").astype("Int64")
        posts["_message_id"] = posts["message_id"]
        return posts

    msgs_path = output_dir / "messages_fitness.parquet"
    if not msgs_path.exists():
        raise FileNotFoundError(
            "posts_metadata.parquet não contém message_id/_message_id e "
            "messages_fitness.parquet não foi encontrado para reconstrução."
        )

    msgs_raw = pd.read_parquet(msgs_path)[["message_id"]].reset_index(drop=True)
    if len(msgs_raw) != len(posts):
        raise ValueError(
            "Não foi possível alinhar posts_metadata.parquet com messages_fitness.parquet. "
            "Regenere a preparação de dados com IDs explícitos."
        )

    posts["message_id"] = pd.to_numeric(msgs_raw["message_id"], errors="coerce").astype("Int64")
    posts["_message_id"] = posts["message_id"]
    return posts


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
        help="Caminho opcional do dataset para registrar proveniência",
    )
    parser.add_argument(
        "--scale-factor",
        type=str,
        default=None,
        help="Scale factor opcional para registrar proveniência",
    )
    parser.add_argument(
        "--dados-dir",
        type=str,
        default=None,
        help="Override opcional do diretório de dados preparados",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Override opcional do diretório de extração",
    )
    parser.add_argument(
        "--splits-dir",
        type=str,
        default=None,
        help="Override opcional do diretório de saída dos splits",
    )
    args = parser.parse_args()

    validar_proporcoes(args.train, args.val, args.test)

    context = dataset_context(
        dataset_key=args.dataset_key,
        dataset_path=args.dataset_path,
        scale_factor=args.scale_factor,
    )
    dados_dir = Path(args.dados_dir) if args.dados_dir else context.dados_dir
    if not dados_dir.is_absolute():
        dados_dir = (ROOT / dados_dir).resolve()
    output_dir = Path(args.output_dir) if args.output_dir else context.output_dir
    if not output_dir.is_absolute():
        output_dir = (ROOT / output_dir).resolve()
    splits_dir = Path(args.splits_dir) if args.splits_dir else context.splits_dir
    if not splits_dir.is_absolute():
        splits_dir = (ROOT / splits_dir).resolve()

    print("=== Divisão do dataset ===")
    print(f"  Namespace  : {context.dataset_key or 'legado'}")
    print(f"  Dados      : {dados_dir}")
    print(f"  Extração   : {output_dir}")
    print(f"  Splits     : {splits_dir}")
    print(f"  Proporções : treino={args.train:.0%}  validação={args.val:.0%}  teste={args.test:.0%}")
    print(f"  Seed       : {args.seed}\n")

    # --- Carregar posts ---
    posts_path = dados_dir / "posts_metadata.parquet"
    if not posts_path.exists():
        print("ERRO: posts_metadata.parquet não encontrado.")
        print("Execute primeiro: python treinamento/preparacao_dados.py")
        sys.exit(1)

    posts = pd.read_parquet(posts_path)
    posts["tags_fitness"] = posts["tags_fitness"].apply(_parse_tags)
    posts = garantir_message_ids(posts, output_dir)
    # Índice posicional para rastrear a posição de cada post após o embaralhamento
    posts["post_idx_original"] = posts.index
    total = len(posts)
    print(f"  {total} posts carregados de {posts_path.name}")

    # --- Carregar interações e grafo social ---
    inter_path = output_dir / "interactions_fitness.parquet"
    social_path = output_dir / "user_social_graph.parquet"
    has_interactions = inter_path.exists()
    has_social = social_path.exists()

    if has_interactions:
        interactions = pd.read_parquet(inter_path)
        print(f"  {len(interactions)} interações carregadas de {inter_path.name}")
    else:
        print("  [AVISO] interactions_fitness.parquet não encontrado — splits de interações ignorados")

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
    splits_dir.mkdir(parents=True, exist_ok=True)

    splits_posts = [("train", train), ("val", val), ("test", test)]
    for nome, df in splits_posts:
        caminho = splits_dir / f"{nome}_posts.parquet"
        # Mantém colunas internas para rastreabilidade dos experimentos e benchmark.
        df.to_parquet(caminho, index=True)
        print(f"  {nome}_posts.parquet salvo: {len(df)} posts")

    train_inter_count = 0
    social_scores_train_count = 0
    if has_interactions:
        for nome, df in splits_posts:
            msg_ids = set(df["_message_id"].tolist())
            df_inter = interactions[interactions["message_id"].isin(msg_ids)].copy()
            caminho = splits_dir / f"{nome}_interactions.parquet"
            df_inter.to_parquet(caminho, index=False)
            print(f"  {nome}_interactions.parquet salvo: {len(df_inter)} interações")
            if nome == "train":
                train_inter_count = int(len(df_inter))

    # --- Recalcular co-ocorrência SÓ com treino (evita data leakage) ---
    print("\nRecalculando tag_cooccurrence com dados de treino...")
    cooc_treino = recalcular_cooccurrence(train)
    caminho_cooc = splits_dir / "train_tag_cooccurrence.parquet"
    cooc_treino.to_parquet(caminho_cooc, index=False)
    print(f"  train_tag_cooccurrence.parquet salvo: {len(cooc_treino)} pares de tags")

    # --- Recalcular social scores SÓ com treino (evita data leakage) ---
    if has_interactions and has_social and "_message_id" in train.columns:
        print("\nRecalculando social_scores com dados de treino...")
        train_msg_ids = set(train["_message_id"].tolist())
        train_inter = interactions[interactions["message_id"].isin(train_msg_ids)].copy()
        social_scores_treino = recalcular_social_scores(train, train_inter, social_graph)
        caminho_social_scores = splits_dir / "train_social_scores.parquet"
        social_scores_treino.to_parquet(caminho_social_scores, index=True)
        score_medio = social_scores_treino["social_score"].mean()
        social_scores_train_count = int(len(social_scores_treino))
        print(f"  train_social_scores.parquet salvo: {len(social_scores_treino)} posts")
        print(f"  Score médio de influência social: {score_medio:.4f}")
    else:
        print("\n[AVISO] Grafo social ou interações ausentes — train_social_scores.parquet não gerado.")

    # --- Resumo final ---
    imprimir_resumo(total, train, val, test, args.train, args.val, args.seed)
    split_cfg_payload = normalize_split_config(
        {
            "train": args.train,
            "val": args.val,
            "test": args.test,
            "seed": args.seed,
        }
    )
    manifest = build_stage_manifest(
        stage="divisao_dataset",
        context=context,
        extra={
            "dados_dir": rel_path(dados_dir),
            "output_dir": rel_path(output_dir),
            "splits_dir": rel_path(splits_dir),
            "split_config": split_cfg_payload,
            "split_signature": split_signature(split_cfg_payload),
            "data_contract": {
                "post_id_column": "_message_id",
                "interaction_message_column": "message_id",
                "timestamp_unit": "ms",
            },
            "summary": {
                "total_posts": int(total),
                "train_posts": int(len(train)),
                "val_posts": int(len(val)),
                "test_posts": int(len(test)),
                "train_interactions": train_inter_count,
                "social_scores_train": social_scores_train_count,
            },
        },
    )
    write_manifest(splits_dir, manifest)
    print(f"Splits salvos em: {splits_dir}")


if __name__ == "__main__":
    main()
