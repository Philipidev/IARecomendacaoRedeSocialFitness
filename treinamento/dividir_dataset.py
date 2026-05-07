"""
Divisão do dataset em conjuntos de treino, validação e teste.

Suporta três estratégias:

  * temporal_global (padrão recomendado): ordena TODAS as interações por
    timestamp e divide cronologicamente em treino/val/teste. Posts ficam
    inteiros no catálogo para refletir o cenário real de produção.

  * leave_last_k: para cada usuário, separa as últimas K interações para
    teste, as K anteriores para validação e o restante para treino.

  * random: split aleatório de POSTS (estratégia legada). Mantida por
    retrocompatibilidade — não recomendada para avaliação de recsys com
    sinal temporal.

Em todos os modos os posts permanecem disponíveis no catálogo. O que muda
é como as INTERAÇÕES são particionadas (ground truth) e quais estatísticas
derivadas (tag_cooccurrence, social_scores) são recalculadas.

Uso:
    python treinamento/dividir_dataset.py
    python treinamento/dividir_dataset.py --strategy temporal_global
    python treinamento/dividir_dataset.py --strategy leave_last_k --leave-last-k 2
    python treinamento/dividir_dataset.py --strategy random --train 0.8 --val 0.1 --test 0.1
"""

from __future__ import annotations

import argparse
import ast
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dataset_context import build_stage_manifest, dataset_context, rel_path, write_manifest
from pipeline_contracts import (
    DEFAULT_SPLIT_STRATEGY,
    VALID_SPLIT_STRATEGIES,
    detect_time_column,
    normalize_split_config,
    split_signature,
    timestamp_to_ms,
)

PROPORCAO_TREINO_PADRAO = 0.70
PROPORCAO_VAL_PADRAO = 0.15
PROPORCAO_TESTE_PADRAO = 0.15
SEED_PADRAO = 42
LEAVE_LAST_K_PADRAO = 1


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
    n_val = int(total * p_val)
    n_teste = int(total * (1.0 - p_treino - p_val))
    n_treino = total - n_val - n_teste
    return n_treino, n_val


def dividir_posts_aleatorio(
    posts: pd.DataFrame,
    p_treino: float,
    p_val: float,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    shuffled = posts.sample(frac=1, random_state=seed).reset_index(drop=True)
    total = len(shuffled)
    n_treino, n_val = calcular_cortes(total, p_treino, p_val)

    train = shuffled.iloc[:n_treino].copy()
    val = shuffled.iloc[n_treino : n_treino + n_val].copy()
    test = shuffled.iloc[n_treino + n_val :].copy()
    return train, val, test


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


def _normalizar_interacoes(interactions: pd.DataFrame) -> tuple[pd.DataFrame, str]:
    tempo_col = detect_time_column(interactions)
    if tempo_col is None:
        raise ValueError(
            "interactions_fitness.parquet não possui coluna temporal reconhecida; "
            "split temporal exige timestamp."
        )

    df = interactions.copy()
    df["__ts_ms"] = df[tempo_col].apply(timestamp_to_ms)
    if df["__ts_ms"].isna().all():
        df["__ts_ms"] = np.arange(len(df), dtype=np.int64)
    df["__ts_ms"] = pd.to_numeric(df["__ts_ms"], errors="coerce")
    df = df.dropna(subset=["__ts_ms"]).copy()
    df["__ts_ms"] = df["__ts_ms"].astype(np.int64)
    df["message_id"] = pd.to_numeric(df["message_id"], errors="coerce").astype("Int64")
    df["user_id"] = pd.to_numeric(df["user_id"], errors="coerce").astype("Int64")
    df = df.dropna(subset=["message_id", "user_id"]).copy()
    df["message_id"] = df["message_id"].astype("int64")
    df["user_id"] = df["user_id"].astype("int64")
    return df, tempo_col


def split_temporal_global(
    interactions: pd.DataFrame,
    p_treino: float,
    p_val: float,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    """
    Ordena cronologicamente todas as interações e corta por percentil temporal.
    """
    if interactions.empty:
        empty = interactions.iloc[0:0].copy()
        return empty, empty, empty, {"cut_train_val_ms": None, "cut_val_test_ms": None}

    sorted_df = interactions.sort_values("__ts_ms", kind="mergesort").reset_index(drop=True)
    total = len(sorted_df)
    n_train, n_val = calcular_cortes(total, p_treino, p_val)

    train = sorted_df.iloc[:n_train].copy()
    val = sorted_df.iloc[n_train : n_train + n_val].copy()
    test = sorted_df.iloc[n_train + n_val :].copy()

    cut_train_val = int(train["__ts_ms"].max()) if not train.empty else None
    cut_val_test = int(val["__ts_ms"].max()) if not val.empty else cut_train_val

    diagnostics = {
        "cut_train_val_ms": cut_train_val,
        "cut_val_test_ms": cut_val_test,
        "min_ts_ms": int(sorted_df["__ts_ms"].min()),
        "max_ts_ms": int(sorted_df["__ts_ms"].max()),
    }
    return train, val, test, diagnostics


def split_leave_last_k(
    interactions: pd.DataFrame,
    leave_last_k: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    """
    Para cada usuário, separa as últimas K interações para teste, as K anteriores
    para validação e o restante para treino.
    """
    leave_last_k = max(1, int(leave_last_k))
    if interactions.empty:
        empty = interactions.iloc[0:0].copy()
        return empty, empty, empty, {"leave_last_k": leave_last_k, "users_with_test": 0}

    train_idx: list = []
    val_idx: list = []
    test_idx: list = []
    users_with_test = 0
    sorted_df = interactions.sort_values(["user_id", "__ts_ms"], kind="mergesort")

    for _, group in sorted_df.groupby("user_id", sort=False):
        idxs = list(group.index)
        n = len(idxs)
        if n <= leave_last_k:
            train_idx.extend(idxs)
            continue
        n_test = leave_last_k
        n_val = leave_last_k if n > 2 * leave_last_k else 0
        n_train = n - n_test - n_val
        train_idx.extend(idxs[:n_train])
        if n_val:
            val_idx.extend(idxs[n_train : n_train + n_val])
        test_idx.extend(idxs[n_train + n_val :])
        users_with_test += 1

    train = sorted_df.loc[train_idx].copy()
    val = sorted_df.loc[val_idx].copy()
    test = sorted_df.loc[test_idx].copy()

    diagnostics = {
        "leave_last_k": leave_last_k,
        "users_with_test": users_with_test,
        "users_total": int(sorted_df["user_id"].nunique()),
    }
    return train, val, test, diagnostics


def recalcular_cooccurrence_de_posts(posts: pd.DataFrame) -> pd.DataFrame:
    """Co-ocorrência de tags em posts. Tags são metadados estáticos (sem leakage)."""
    contagem: dict[tuple[str, str], int] = defaultdict(int)

    for tags in posts["tags_fitness"]:
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
    posts_catalog: pd.DataFrame,
    train_interactions: pd.DataFrame,
    social_graph: pd.DataFrame,
) -> pd.DataFrame:
    """Social scores derivados apenas de interações de treino (evita leakage)."""
    if social_graph.empty or train_interactions.empty:
        return pd.DataFrame(
            {"social_score": np.zeros(len(posts_catalog), dtype=np.float32)}
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
        [msg_score.get(int(mid), 0.0) for mid in posts_catalog["_message_id"].values],
        dtype=np.float32,
    )

    max_score = scores.max()
    if max_score > 0:
        scores /= max_score

    return pd.DataFrame({"social_score": scores})


def imprimir_resumo_proporcoes(
    total: int,
    train: pd.DataFrame,
    val: pd.DataFrame,
    test: pd.DataFrame,
    label: str,
) -> None:
    print(f"\n{'='*52}")
    print(f"  Resumo de {label}: {total} registros")
    print(f"{'='*52}")
    for nome, df in [("treino", train), ("validação", val), ("teste", test)]:
        pct_real = (len(df) / total * 100) if total else 0.0
        print(f"  {nome:<12} {len(df):>8}   ({pct_real:.1f}%)")
    print(f"{'='*52}\n")


def _ts_ms_para_iso(ts_ms: int | None) -> str | None:
    if ts_ms is None:
        return None
    try:
        return pd.Timestamp(int(ts_ms), unit="ms", tz="UTC").isoformat()
    except Exception:
        return None


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Divide o dataset em treino, validação e teste.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemplos:
  python treinamento/dividir_dataset.py
  python treinamento/dividir_dataset.py --strategy temporal_global
  python treinamento/dividir_dataset.py --strategy leave_last_k --leave-last-k 1
  python treinamento/dividir_dataset.py --strategy random --train 0.8 --val 0.1 --test 0.1
        """,
    )
    parser.add_argument(
        "--strategy",
        type=str,
        default=DEFAULT_SPLIT_STRATEGY,
        choices=VALID_SPLIT_STRATEGIES,
        help=(
            "Estratégia de divisão. temporal_global (recomendado), "
            "leave_last_k ou random (legado)."
        ),
    )
    parser.add_argument(
        "--leave-last-k",
        type=int,
        default=LEAVE_LAST_K_PADRAO,
        help="Para leave_last_k: quantas interações por usuário ficam em teste/val.",
    )
    parser.add_argument("--train", type=float, default=PROPORCAO_TREINO_PADRAO)
    parser.add_argument("--val", type=float, default=PROPORCAO_VAL_PADRAO)
    parser.add_argument("--test", type=float, default=PROPORCAO_TESTE_PADRAO)
    parser.add_argument("--seed", type=int, default=SEED_PADRAO)
    parser.add_argument("--dataset-key", type=str, default=None)
    parser.add_argument("--dataset-path", type=str, default=None)
    parser.add_argument("--scale-factor", type=str, default=None)
    parser.add_argument("--dados-dir", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--splits-dir", type=str, default=None)
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
    print(f"  Estratégia : {args.strategy}")
    print(f"  Dados      : {dados_dir}")
    print(f"  Extração   : {output_dir}")
    print(f"  Splits     : {splits_dir}")
    if args.strategy == "random":
        print(
            f"  Proporções : treino={args.train:.0%}  "
            f"val={args.val:.0%}  teste={args.test:.0%} (split de POSTS)"
        )
    elif args.strategy == "temporal_global":
        print(
            f"  Proporções : treino={args.train:.0%}  "
            f"val={args.val:.0%}  teste={args.test:.0%} (split temporal de interações)"
        )
    else:
        print(f"  Leave last K: {args.leave_last_k} interações por usuário em teste/val")
    print(f"  Seed       : {args.seed}\n")

    posts_path = dados_dir / "posts_metadata.parquet"
    if not posts_path.exists():
        print("ERRO: posts_metadata.parquet não encontrado.")
        print("Execute primeiro: python treinamento/preparacao_dados.py")
        sys.exit(1)

    posts = pd.read_parquet(posts_path)
    posts["tags_fitness"] = posts["tags_fitness"].apply(_parse_tags)
    posts = garantir_message_ids(posts, output_dir)
    posts["post_idx_original"] = posts.index
    total_posts = len(posts)
    print(f"  {total_posts} posts carregados de {posts_path.name}")

    inter_path = output_dir / "interactions_fitness.parquet"
    social_path = output_dir / "user_social_graph.parquet"
    has_interactions = inter_path.exists()
    has_social = social_path.exists()

    interactions_norm = pd.DataFrame()
    interactions_raw = pd.DataFrame()
    if has_interactions:
        interactions_raw = pd.read_parquet(inter_path)
        print(f"  {len(interactions_raw)} interações carregadas de {inter_path.name}")
        try:
            interactions_norm, _ = _normalizar_interacoes(interactions_raw)
        except ValueError as exc:
            if args.strategy != "random":
                print(f"ERRO: {exc}")
                sys.exit(1)
            print(f"  [AVISO] {exc} (estratégia random tolera ausência)")
    else:
        print("  [AVISO] interactions_fitness.parquet não encontrado")
        if args.strategy != "random":
            print("ERRO: split temporal/leave_last_k exige interactions_fitness.parquet")
            sys.exit(1)

    social_graph = pd.DataFrame()
    if has_social:
        social_graph = pd.read_parquet(social_path)
        print(f"  {len(social_graph)} arestas carregadas de {social_path.name}")

    splits_dir.mkdir(parents=True, exist_ok=True)

    train_inter = pd.DataFrame()
    val_inter = pd.DataFrame()
    test_inter = pd.DataFrame()
    train_posts_df = posts
    val_posts_df = pd.DataFrame()
    test_posts_df = pd.DataFrame()
    strategy_diagnostics: dict = {}

    if args.strategy == "random":
        print("\nDividindo posts (estratégia random — legada)...")
        train_posts_df, val_posts_df, test_posts_df = dividir_posts_aleatorio(
            posts, args.train, args.val, args.seed
        )
        imprimir_resumo_proporcoes(
            total_posts, train_posts_df, val_posts_df, test_posts_df, "posts"
        )
        for nome, df in [
            ("train", train_posts_df),
            ("val", val_posts_df),
            ("test", test_posts_df),
        ]:
            df.to_parquet(splits_dir / f"{nome}_posts.parquet", index=True)
            print(f"  {nome}_posts.parquet salvo: {len(df)} posts")

        if has_interactions and not interactions_raw.empty:
            for nome, df in [
                ("train", train_posts_df),
                ("val", val_posts_df),
                ("test", test_posts_df),
            ]:
                msg_ids = set(df["_message_id"].dropna().astype("int64").tolist())
                df_inter = interactions_raw[
                    interactions_raw["message_id"].isin(msg_ids)
                ].copy()
                df_inter.to_parquet(splits_dir / f"{nome}_interactions.parquet", index=False)
                print(f"  {nome}_interactions.parquet salvo: {len(df_inter)}")
                if nome == "train":
                    train_inter = df_inter
                elif nome == "val":
                    val_inter = df_inter
                else:
                    test_inter = df_inter

        cooc = recalcular_cooccurrence_de_posts(train_posts_df)
    else:
        if args.strategy == "temporal_global":
            print("\nDividindo INTERAÇÕES por timestamp (temporal_global)...")
            train_inter, val_inter, test_inter, strategy_diagnostics = (
                split_temporal_global(interactions_norm, args.train, args.val)
            )
        else:
            print("\nDividindo INTERAÇÕES por usuário (leave_last_k)...")
            train_inter, val_inter, test_inter, strategy_diagnostics = (
                split_leave_last_k(interactions_norm, args.leave_last_k)
            )

        imprimir_resumo_proporcoes(
            len(interactions_norm),
            train_inter,
            val_inter,
            test_inter,
            "interações",
        )

        # Catálogo de posts permanece inteiro: refletimos o cenário real em que
        # o modelo recomenda dentre todos os posts conhecidos. As cópias por
        # split mantêm o contrato dos consumidores (treinar.py, treinar_ltr.py).
        for nome in ("train", "val", "test"):
            posts.to_parquet(splits_dir / f"{nome}_posts.parquet", index=True)
            print(f"  {nome}_posts.parquet salvo: {len(posts)} posts (catálogo completo)")
        train_posts_df = posts
        val_posts_df = posts
        test_posts_df = posts

        for nome, df in [("train", train_inter), ("val", val_inter), ("test", test_inter)]:
            df_to_save = df.drop(columns=["__ts_ms"], errors="ignore")
            df_to_save.to_parquet(splits_dir / f"{nome}_interactions.parquet", index=False)
            print(f"  {nome}_interactions.parquet salvo: {len(df_to_save)}")

        # Co-ocorrência: tags são metadados estáticos. Pode usar o catálogo
        # inteiro sem leakage (não envolve interações).
        cooc = recalcular_cooccurrence_de_posts(posts)

    caminho_cooc = splits_dir / "train_tag_cooccurrence.parquet"
    cooc.to_parquet(caminho_cooc, index=False)
    print(f"\n  train_tag_cooccurrence.parquet salvo: {len(cooc)} pares")

    social_scores_train_count = 0
    if has_interactions and has_social and not train_inter.empty:
        print("\nRecalculando social_scores com interações de treino...")
        train_inter_for_social = train_inter.drop(columns=["__ts_ms"], errors="ignore")
        social_df = recalcular_social_scores(posts, train_inter_for_social, social_graph)
        social_df.to_parquet(splits_dir / "train_social_scores.parquet", index=True)
        social_scores_train_count = int(len(social_df))
        print(
            f"  train_social_scores.parquet salvo: {social_scores_train_count} posts "
            f"(score médio={float(social_df['social_score'].mean()):.4f})"
        )
    else:
        print("\n[AVISO] train_social_scores.parquet não gerado (faltam dados).")

    split_cfg_payload = normalize_split_config(
        {
            "train": args.train,
            "val": args.val,
            "test": args.test,
            "seed": args.seed,
            "strategy": args.strategy,
            "leave_last_k": args.leave_last_k,
        }
    )
    cut_train_val = strategy_diagnostics.get("cut_train_val_ms")
    cut_val_test = strategy_diagnostics.get("cut_val_test_ms")
    manifest = build_stage_manifest(
        stage="divisao_dataset",
        context=context,
        extra={
            "dados_dir": rel_path(dados_dir),
            "output_dir": rel_path(output_dir),
            "splits_dir": rel_path(splits_dir),
            "split_config": split_cfg_payload,
            "split_signature": split_signature(split_cfg_payload),
            "split_strategy": args.strategy,
            "temporal_cuts": {
                "cut_train_val_ms": cut_train_val,
                "cut_val_test_ms": cut_val_test,
                "cut_train_val_iso": _ts_ms_para_iso(cut_train_val),
                "cut_val_test_iso": _ts_ms_para_iso(cut_val_test),
            }
            if args.strategy == "temporal_global"
            else None,
            "leave_last_k": args.leave_last_k if args.strategy == "leave_last_k" else None,
            "data_contract": {
                "post_id_column": "_message_id",
                "interaction_message_column": "message_id",
                "timestamp_unit": "ms",
            },
            "summary": {
                "total_posts": int(total_posts),
                "train_posts": int(len(train_posts_df)),
                "val_posts": int(len(val_posts_df)) if isinstance(val_posts_df, pd.DataFrame) else 0,
                "test_posts": int(len(test_posts_df)) if isinstance(test_posts_df, pd.DataFrame) else 0,
                "train_interactions": int(len(train_inter)),
                "val_interactions": int(len(val_inter)),
                "test_interactions": int(len(test_inter)),
                "social_scores_train": social_scores_train_count,
            },
            "strategy_diagnostics": strategy_diagnostics,
        },
    )
    write_manifest(splits_dir, manifest)
    print(f"\nSplits salvos em: {splits_dir}")


if __name__ == "__main__":
    main()
