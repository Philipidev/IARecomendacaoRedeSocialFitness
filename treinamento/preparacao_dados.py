"""
Preparação dos dados para treinamento da IA de recomendação.

Lê os parquets gerados pelo pipeline de extração e produz:
  - posts_metadata.parquet     : posts com apenas valores (sem IDs na interface de saída)
  - interacoes_por_tag.parquet : contagem de interações por tag (para peso de popularidade)
  - social_scores.parquet      : score de influência social por post (grau dos usuários que interagiram)
  - user_tag_profile.parquet   : afinidade usuário-tag (interesses + histórico + vizinhos)
  - tag_lista.txt              : lista canônica de todas as tags fitness conhecidas

Uso:
    python treinamento/preparacao_dados.py
"""

from __future__ import annotations

import argparse
import ast
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
TREINAMENTO_DIR = ROOT / "treinamento"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dataset_context import build_stage_manifest, dataset_context, rel_path, write_manifest

EVENTO_PESO = {
    "like": 1.0,
    "reply": 1.2,
    "create": 1.5,
}
LAMBDA_RECENCIA_USUARIO = 0.03
MS_POR_DIA = 86_400_000


def _parse_tags(value) -> list[str]:
    """Normaliza o campo tags_fitness independente do tipo armazenado."""
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


def carregar_parquets(output_dir: Path) -> dict[str, pd.DataFrame]:
    nomes = [
        "messages_fitness",
        "tags_fitness",
        "tag_cooccurrence",
        "interactions_fitness",
        "user_interests_fitness",
        "user_social_graph",
    ]
    dados = {}
    for nome in nomes:
        caminho = output_dir / f"{nome}.parquet"
        if not caminho.exists():
            print(f"[AVISO] {caminho} não encontrado — pulando.")
            continue
        df = pd.read_parquet(caminho)
        dados[nome] = df
        print(f"  {nome}: {df.shape[0]} linhas, colunas: {list(df.columns)}")
    return dados


def preparar_posts(messages: pd.DataFrame) -> pd.DataFrame:
    """
    Constrói o DataFrame de posts para o modelo.
    Preserva explicitamente message_id para evitar dependência de alinhamento posicional
    com messages_fitness.parquet.
    """
    df = messages.copy()

    df["tags_fitness"] = df["tags_fitness"].apply(_parse_tags)

    if "message_id" in df.columns:
        df["message_id"] = pd.to_numeric(df["message_id"], errors="coerce").astype("Int64")
        df["_message_id"] = df["message_id"]

    # Contrato temporal único do pipeline: timestamps em ms.
    df["creation_date"] = pd.to_numeric(df["creation_date"], errors="coerce")
    df["creation_date_dt"] = pd.to_datetime(df["creation_date"], unit="ms", utc=True)
    df["creation_date_iso"] = df["creation_date_dt"].dt.strftime("%Y-%m-%dT%H:%M:%SZ")

    # Índice posicional usado internamente pelo modelo; o identificador canônico continua em _message_id.
    df = df.reset_index(drop=True)
    df.index.name = "post_idx"

    colunas_saida = [
        "_message_id",
        "message_id",
        "message_type",
        "creation_date",
        "creation_date_iso",
        "content_length",
        "language",
        "tags_fitness",
    ]
    return df[colunas_saida]


def calcular_popularidade_tags(interactions: pd.DataFrame) -> pd.DataFrame:
    """
    Conta quantas interações cada tag recebeu.
    Usado como sinal de popularidade no score final.
    """
    linhas = []
    for _, row in interactions.iterrows():
        tags = _parse_tags(row["tags_fitness"])
        evento = row["event_type"]
        for tag in tags:
            linhas.append({"tag_name": tag, "event_type": evento})

    if not linhas:
        return pd.DataFrame(columns=["tag_name", "total_interacoes"])

    df_flat = pd.DataFrame(linhas)
    pop = df_flat.groupby("tag_name").size().reset_index(name="total_interacoes")
    return pop.sort_values("total_interacoes", ascending=False)


def calcular_scores_sociais(
    messages: pd.DataFrame,
    interactions: pd.DataFrame,
    social_graph: pd.DataFrame,
) -> pd.DataFrame:
    """
    Calcula o Social Influence Score para cada post.

    Para cada post, soma o grau (número de conexões no grafo social) de cada
    usuário que interagiu com ele. Posts interagidos por usuários altamente
    conectados (influenciadores) recebem score mais alto.

    Parâmetros
    ----------
    messages : DataFrame com coluna message_id (linha i == post_idx i)
    interactions : DataFrame com colunas user_id e message_id
    social_graph : DataFrame com colunas user_id e friend_id

    Retorna
    -------
    DataFrame com coluna social_score, indexado por post_idx (0-based).
    """
    # Grau de cada usuário = aparições como user_id + aparições como friend_id
    degree_as_user = social_graph["user_id"].value_counts()
    degree_as_friend = social_graph["friend_id"].value_counts()
    degree_map: dict = degree_as_user.add(degree_as_friend, fill_value=0).to_dict()

    # Score por message_id = soma dos graus dos usuários que interagiram
    msg_score: dict = (
        interactions.groupby("message_id")["user_id"]
        .apply(lambda uids: float(sum(degree_map.get(int(u), 0) for u in uids)))
        .to_dict()
    )

    # Alinha com a ordem dos posts (row i de messages → post_idx i)
    messages_reset = messages.reset_index(drop=True)
    scores = np.array(
        [msg_score.get(int(mid), 0.0) for mid in messages_reset["message_id"].values],
        dtype=np.float32,
    )

    max_score = scores.max()
    if max_score > 0:
        scores /= max_score

    return pd.DataFrame({"social_score": scores})


def construir_perfis_usuario(
    interactions: pd.DataFrame,
    interests: pd.DataFrame,
    social_graph: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """
    Constrói score de afinidade por (user_id, tag_name) combinando:
      1) interesses explícitos,
      2) histórico de interações com recência,
      3) sinais sociais dos vizinhos (grafo de amizades).
    """

    # ------------------------------
    # Interesses explícitos
    # ------------------------------
    if interests is not None and not interests.empty:
        explicit_df = (
            interests.groupby(["user_id", "tag_name"]).size().reset_index(name="explicit_score")
        )
        explicit_df["explicit_score"] = explicit_df["explicit_score"].astype(np.float32)
    else:
        explicit_df = pd.DataFrame(columns=["user_id", "tag_name", "explicit_score"])

    # ------------------------------
    # Interações (com recência + tipo de evento)
    # ------------------------------
    inter = interactions.copy()
    inter["timestamp"] = pd.to_numeric(inter["timestamp"], errors="coerce")
    ts_max = inter["timestamp"].max()

    linhas_inter = []
    for _, row in inter.iterrows():
        tags = _parse_tags(row.get("tags_fitness"))
        if not tags:
            continue

        idade_dias = 0.0
        if pd.notna(ts_max) and pd.notna(row["timestamp"]):
            idade_dias = max(0.0, float(ts_max - row["timestamp"]) / MS_POR_DIA)

        peso_evento = EVENTO_PESO.get(str(row.get("event_type", "")).lower(), 1.0)
        peso_recencia = float(np.exp(-LAMBDA_RECENCIA_USUARIO * idade_dias))
        peso = peso_evento * peso_recencia

        for tag in tags:
            linhas_inter.append(
                {
                    "user_id": int(row["user_id"]),
                    "tag_name": str(tag),
                    "interaction_score": peso,
                }
            )

    if linhas_inter:
        interaction_df = pd.DataFrame(linhas_inter)
        interaction_df = (
            interaction_df.groupby(["user_id", "tag_name"], as_index=False)["interaction_score"]
            .sum()
            .astype({"interaction_score": np.float32})
        )
    else:
        interaction_df = pd.DataFrame(columns=["user_id", "tag_name", "interaction_score"])

    # ------------------------------
    # Sinal social dos vizinhos
    # ------------------------------
    neighbor_df = pd.DataFrame(columns=["user_id", "tag_name", "neighbor_score"])
    if social_graph is not None and not social_graph.empty and not interaction_df.empty:
        edges = social_graph[["user_id", "friend_id"]].dropna().copy()
        edges = edges.astype({"user_id": "int64", "friend_id": "int64"})

        # Grafo não-direcionado: adiciona arestas invertidas
        edges_rev = edges.rename(columns={"user_id": "friend_id", "friend_id": "user_id"})
        edges_undir = pd.concat([edges, edges_rev], ignore_index=True).drop_duplicates()
        edges_undir = edges_undir.rename(columns={"friend_id": "neighbor_id"})

        source_scores = interaction_df.rename(
            columns={"user_id": "neighbor_id", "interaction_score": "neighbor_score"}
        )

        neighbor_df = edges_undir.merge(source_scores, on="neighbor_id", how="inner")
        neighbor_df = (
            neighbor_df.groupby(["user_id", "tag_name"], as_index=False)["neighbor_score"]
            .sum()
            .astype({"neighbor_score": np.float32})
        )

    # ------------------------------
    # Combina e normaliza por usuário
    # ------------------------------
    perfil = explicit_df.merge(interaction_df, on=["user_id", "tag_name"], how="outer")
    perfil = perfil.merge(neighbor_df, on=["user_id", "tag_name"], how="outer")
    perfil = perfil.fillna(0.0)

    for col in ["explicit_score", "interaction_score", "neighbor_score"]:
        max_by_user = perfil.groupby("user_id")[col].transform("max")
        max_by_user = max_by_user.replace(0.0, 1.0)
        perfil[col] = (perfil[col] / max_by_user).astype(np.float32)

    perfil["user_tag_affinity"] = (
        0.4 * perfil["interaction_score"]
        + 0.4 * perfil["explicit_score"]
        + 0.2 * perfil["neighbor_score"]
    ).astype(np.float32)

    return perfil.sort_values(["user_id", "user_tag_affinity"], ascending=[True, False]).reset_index(drop=True)


def salvar_metricas_txt(dados: dict[str, pd.DataFrame], dados_dir: Path) -> None:
    """Exporta um .txt por métrica relevante — uma linha por valor único."""

    def _write(nome: str, linhas: list[str]) -> None:
        caminho = dados_dir / nome
        caminho.write_text("\n".join(linhas), encoding="utf-8")
        print(f"  {nome}: {len(linhas)} valores salvos em {caminho}")

    if "interactions_fitness" in dados:
        vals = sorted(dados["interactions_fitness"]["event_type"].dropna().unique())
        _write("event_type_lista.txt", [str(v) for v in vals])

    if "messages_fitness" in dados:
        langs = sorted(dados["messages_fitness"]["language"].dropna().unique())
        _write("language_lista.txt", [str(v) for v in langs])

        tipos = sorted(dados["messages_fitness"]["message_type"].dropna().unique())
        _write("message_type_lista.txt", [str(v) for v in tipos])

    if "interactions_fitness" in dados:
        uids = sorted(dados["interactions_fitness"]["user_id"].dropna().unique())
        _write("user_id_lista.txt", [str(int(v)) for v in uids])

    if "tag_cooccurrence" in dados:
        df = dados["tag_cooccurrence"].sort_values("cooccurrences", ascending=False)
        linhas = [f"{r.tag_a}|{r.tag_b}|{r.cooccurrences}" for r in df.itertuples()]
        _write("tag_cooccurrence_pares_lista.txt", linhas)


def salvar_tag_lista(tags_fitness: pd.DataFrame, dados_dir: Path) -> None:
    nomes = sorted(tags_fitness["tag_name"].dropna().unique().tolist())
    caminho = dados_dir / "tag_lista.txt"
    caminho.write_text("\n".join(nomes), encoding="utf-8")
    print(f"  tag_lista.txt: {len(nomes)} tags salvas em {caminho}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepara artefatos derivados para treino.")
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
        "--output-dir",
        type=str,
        default=None,
        help="Override opcional do diretório de parquets extraídos",
    )
    parser.add_argument(
        "--dados-dir",
        type=str,
        default=None,
        help="Override opcional do diretório de saída dos dados preparados",
    )
    args = parser.parse_args()

    context = dataset_context(
        dataset_key=args.dataset_key,
        dataset_path=args.dataset_path,
        scale_factor=args.scale_factor,
    )
    output_dir = Path(args.output_dir) if args.output_dir else context.output_dir
    if not output_dir.is_absolute():
        output_dir = (ROOT / output_dir).resolve()
    dados_dir = Path(args.dados_dir) if args.dados_dir else context.dados_dir
    if not dados_dir.is_absolute():
        dados_dir = (ROOT / dados_dir).resolve()

    print("=== Preparação de dados ===\n")
    print(f"Namespace ativo : {context.dataset_key or 'legado'}")
    print(f"Input extração  : {output_dir}")
    print(f"Output dados    : {dados_dir}\n")

    dados_dir.mkdir(parents=True, exist_ok=True)

    print("Carregando parquets...")
    dados = carregar_parquets(output_dir)

    if "messages_fitness" not in dados:
        print("ERRO: messages_fitness.parquet não encontrado. Execute o pipeline primeiro.")
        sys.exit(1)

    print("\nPreparando posts...")
    posts = preparar_posts(dados["messages_fitness"])
    caminho_posts = dados_dir / "posts_metadata.parquet"
    posts.to_parquet(caminho_posts, index=True)
    print(f"  posts_metadata.parquet: {len(posts)} posts salvos em {caminho_posts}")

    if "interactions_fitness" in dados:
        print("\nCalculando popularidade de tags...")
        pop = calcular_popularidade_tags(dados["interactions_fitness"])
        caminho_pop = dados_dir / "interacoes_por_tag.parquet"
        pop.to_parquet(caminho_pop, index=False)
        print(f"  interacoes_por_tag.parquet: {len(pop)} tags salvas em {caminho_pop}")

    if (
        "user_social_graph" in dados
        and "interactions_fitness" in dados
        and "messages_fitness" in dados
    ):
        print("\nCalculando scores de influência social...")
        social_scores = calcular_scores_sociais(
            dados["messages_fitness"],
            dados["interactions_fitness"],
            dados["user_social_graph"],
        )
        caminho_social = dados_dir / "social_scores.parquet"
        social_scores.to_parquet(caminho_social, index=True)
        print(
            f"  social_scores.parquet: {len(social_scores)} posts salvos em {caminho_social}"
        )
        print(f"  Score médio de influência social: {social_scores['social_score'].mean():.4f}")
    else:
        print("\n[AVISO] Grafo social ou interações ausentes — social_scores.parquet não gerado.")

    if "tags_fitness" in dados:
        print("\nSalvando lista canônica de tags...")
        salvar_tag_lista(dados["tags_fitness"], dados_dir)

    if "interactions_fitness" in dados and "user_interests_fitness" in dados:
        print("\nConstruindo perfis de afinidade por usuário...")
        user_tag_profile = construir_perfis_usuario(
            dados["interactions_fitness"],
            dados["user_interests_fitness"],
            dados.get("user_social_graph"),
        )
        caminho_user_profile = dados_dir / "user_tag_profile.parquet"
        user_tag_profile.to_parquet(caminho_user_profile, index=False)
        print(
            f"  user_tag_profile.parquet: {len(user_tag_profile)} pares usuário-tag salvos em {caminho_user_profile}"
        )
    else:
        print(
            "\n[AVISO] interactions_fitness ou user_interests_fitness ausentes — "
            "user_tag_profile.parquet não gerado."
        )

    print("\nSalvando métricas como arquivos .txt...")
    salvar_metricas_txt(dados, dados_dir)

    manifest = build_stage_manifest(
        stage="preparacao_dados",
        context=context,
        extra={
            "source_output_dir": rel_path(output_dir),
            "dados_dir": rel_path(dados_dir),
            "data_contract": {
                "post_id_column": "_message_id",
                "timestamp_unit": "ms",
            },
            "artefatos": [
                "posts_metadata.parquet",
                "interacoes_por_tag.parquet",
                "social_scores.parquet",
                "user_tag_profile.parquet",
                "tag_lista.txt",
            ],
            "summary": {
                "posts_metadata": int(len(posts)),
                "posts_with_message_id": int(posts["_message_id"].notna().sum())
                if "_message_id" in posts.columns
                else 0,
                "interacoes_por_tag": int(len(pop)) if "interactions_fitness" in dados else 0,
                "social_scores": int(len(social_scores))
                if (
                    "user_social_graph" in dados
                    and "interactions_fitness" in dados
                    and "messages_fitness" in dados
                )
                else 0,
                "user_tag_profile": int(len(user_tag_profile))
                if "interactions_fitness" in dados and "user_interests_fitness" in dados
                else 0,
            },
        },
    )
    write_manifest(dados_dir, manifest)

    print("\nPreparação concluída.")
    print(f"Artefatos em: {dados_dir}")


if __name__ == "__main__":
    main()
