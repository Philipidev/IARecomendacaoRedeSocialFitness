"""
Preparação dos dados para treinamento da IA de recomendação.

Lê os parquets gerados pelo pipeline de extração e produz:
  - posts_metadata.parquet     : posts com apenas valores (sem IDs na interface de saída)
  - interacoes_por_tag.parquet : contagem de interações por tag (para peso de popularidade)
  - social_scores.parquet      : score de influência social por post (grau dos usuários que interagiram)
  - tag_lista.txt              : lista canônica de todas as tags fitness conhecidas

Uso:
    python treinamento/preparacao_dados.py
"""

from __future__ import annotations

import ast
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = ROOT / "extracao_filtragem" / "output"
TREINAMENTO_DIR = ROOT / "treinamento"
DADOS_DIR = TREINAMENTO_DIR / "dados"


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


def carregar_parquets() -> dict[str, pd.DataFrame]:
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
        caminho = OUTPUT_DIR / f"{nome}.parquet"
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
    Remove message_id da interface de saída; mantém apenas valores semânticos.
    Internamente usa um índice posicional (post_idx) para lookups vetorizados.
    """
    df = messages.copy()

    df["tags_fitness"] = df["tags_fitness"].apply(_parse_tags)

    # Converte timestamp de ms para datetime legível (garante tipo numérico antes)
    df["creation_date"] = pd.to_numeric(df["creation_date"], errors="coerce")
    df["creation_date_dt"] = pd.to_datetime(df["creation_date"], unit="ms", utc=True)
    df["creation_date_iso"] = df["creation_date_dt"].dt.strftime("%Y-%m-%dT%H:%M:%SZ")

    # Índice posicional usado internamente pelo modelo (não é o message_id)
    df = df.reset_index(drop=True)
    df.index.name = "post_idx"

    # Colunas expostas na saída (sem IDs)
    colunas_saida = [
        "message_type",
        "creation_date",        # timestamp ms — usado para time_decay
        "creation_date_iso",    # legível
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
    print("=== Preparação de dados ===\n")

    DADOS_DIR.mkdir(parents=True, exist_ok=True)

    print("Carregando parquets...")
    dados = carregar_parquets()

    if "messages_fitness" not in dados:
        print("ERRO: messages_fitness.parquet não encontrado. Execute o pipeline primeiro.")
        sys.exit(1)

    print("\nPreparando posts...")
    posts = preparar_posts(dados["messages_fitness"])
    caminho_posts = DADOS_DIR / "posts_metadata.parquet"
    posts.to_parquet(caminho_posts, index=True)
    print(f"  posts_metadata.parquet: {len(posts)} posts salvos em {caminho_posts}")

    if "interactions_fitness" in dados:
        print("\nCalculando popularidade de tags...")
        pop = calcular_popularidade_tags(dados["interactions_fitness"])
        caminho_pop = DADOS_DIR / "interacoes_por_tag.parquet"
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
        caminho_social = DADOS_DIR / "social_scores.parquet"
        social_scores.to_parquet(caminho_social, index=True)
        print(
            f"  social_scores.parquet: {len(social_scores)} posts salvos em {caminho_social}"
        )
        print(f"  Score médio de influência social: {social_scores['social_score'].mean():.4f}")
    else:
        print("\n[AVISO] Grafo social ou interações ausentes — social_scores.parquet não gerado.")

    if "tags_fitness" in dados:
        print("\nSalvando lista canônica de tags...")
        salvar_tag_lista(dados["tags_fitness"], DADOS_DIR)

    print("\nSalvando métricas como arquivos .txt...")
    salvar_metricas_txt(dados, DADOS_DIR)

    print("\nPreparação concluída.")
    print(f"Artefatos em: {DADOS_DIR}")


if __name__ == "__main__":
    main()
