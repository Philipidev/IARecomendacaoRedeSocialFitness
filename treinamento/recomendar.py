"""
Módulo de recomendação de posts fitness.

Carrega os artefatos treinados e expõe a função principal:

    recomendar(
        tags,
        timestamp,
        top_k=10,
        excluir_tags_exatas=True,
        peso_popularidade=0.10,
        user_id=None,
    ) -> pd.DataFrame

O score de relevância combina cinco sinais no modo padrão e cinco no modo
personalizado (`user_id` informado com perfil disponível):
  - Similaridade de conteúdo: coseno entre o vetor de tags de entrada e cada post
  - Co-ocorrência de tags: boost para posts que contêm tags relacionadas às de entrada
  - Recência relativa: decaimento exponencial pela distância temporal em dias
  - Influência social: soma dos graus dos usuários que interagiram com o post
  - Quinto sinal:
      * Popularidade (modo padrão): volume histórico de interações nas tags do post
      * Afinidade usuário-item (modo personalizado): interesses explícitos, interações recentes e vizinhos

Entradas:
    tags      : List[str]  — nomes das tags (valores, não IDs)
    timestamp : int        — timestamp em milissegundos do post de referência
    top_k     : int        — número de recomendações
    user_id   : int|None   — usuário alvo para personalização (opcional)

Saída:
    DataFrame com colunas:
        message_type, creation_date_iso, tags_fitness,
        content_length, language, relevance_score

Uso como script (CLI):
    python treinamento/recomendar.py --tags "Born_to_Run,Superunknown" --timestamp 1320000000000
    python treinamento/recomendar.py --tags "Running_Free" --timestamp 1300000000000 --top-k 5 --user-id 123
"""

from __future__ import annotations

import argparse
import ast
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

ROOT = Path(__file__).resolve().parent.parent
DADOS_DIR = ROOT / "treinamento" / "dados"
MODELO_DIR = ROOT / "treinamento" / "modelo"

# Pesos do score padrão
PESO_COSINE = 0.35
PESO_COOC = 0.25
PESO_TIME = 0.15
PESO_SOCIAL = 0.15
PESO_POPULARIDADE = 0.10

# Pesos do score personalizado
PESO_COSINE_PERSONALIZADO = 0.30
PESO_COOC_PERSONALIZADO = 0.20
PESO_USER_AFFINITY = 0.20

# Lambda do decaimento temporal (por dia)
LAMBDA_DECAY = 0.01
MS_POR_DIA = 86_400_000


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


class ModeloRecomendacao:
    """Carrega os artefatos e mantém o estado do modelo em memória."""

    def __init__(self) -> None:
        self._vectorizer = None
        self._post_matrix: np.ndarray | None = None
        self._cooccurrence_map: dict | None = None
        self._popularidade: np.ndarray | None = None
        self._social_scores: np.ndarray | None = None
        self._posts: pd.DataFrame | None = None
        self._user_tag_profile: pd.DataFrame | None = None

    def carregar(self) -> "ModeloRecomendacao":
        """Carrega todos os artefatos do modelo e os posts de referência."""
        for artefato in ["vectorizer.pkl", "post_matrix.npy", "tag_cooccurrence_map.pkl", "popularidade.npy"]:
            caminho = MODELO_DIR / artefato
            if not caminho.exists():
                raise FileNotFoundError(
                    f"Artefato '{artefato}' não encontrado em {MODELO_DIR}.\n"
                    "Execute primeiro:\n"
                    "  python treinamento/treinar.py"
                )

        with open(MODELO_DIR / "vectorizer.pkl", "rb") as f:
            self._vectorizer = pickle.load(f)

        self._post_matrix = np.load(MODELO_DIR / "post_matrix.npy")
        self._popularidade = np.load(MODELO_DIR / "popularidade.npy")

        with open(MODELO_DIR / "tag_cooccurrence_map.pkl", "rb") as f:
            self._cooccurrence_map = pickle.load(f)

        social_path = MODELO_DIR / "social_scores.npy"
        self._social_scores = np.load(social_path) if social_path.exists() else None

        profile_path = DADOS_DIR / "user_tag_profile.parquet"
        self._user_tag_profile = pd.read_parquet(profile_path) if profile_path.exists() else None

        cache_path = MODELO_DIR / "posts_cache.parquet"
        fallback_path = DADOS_DIR / "posts_metadata.parquet"

        if cache_path.exists():
            posts_path = cache_path
        elif fallback_path.exists():
            posts_path = fallback_path
        else:
            raise FileNotFoundError(
                f"Nenhum arquivo de posts encontrado em {MODELO_DIR} nem em {DADOS_DIR}.\n"
                "Execute primeiro:\n"
                "  python treinamento/preparacao_dados.py\n"
                "  python treinamento/treinar.py"
            )

        self._posts = pd.read_parquet(posts_path)
        self._posts["tags_fitness"] = self._posts["tags_fitness"].apply(_parse_tags)
        return self

    def _score_cosine(self, tags_entrada: list[str]) -> np.ndarray:
        vetor_entrada = self._vectorizer.transform([tags_entrada]).astype(np.float32)
        if vetor_entrada.sum() == 0:
            return np.zeros(len(self._posts), dtype=np.float32)
        return cosine_similarity(vetor_entrada, self._post_matrix).flatten().astype(np.float32)

    def _score_cooccurrence(self, tags_entrada: list[str]) -> np.ndarray:
        tags_conhecidas = set(self._vectorizer.classes_)
        boost_por_tag: dict[str, float] = {}

        for tag in tags_entrada:
            for tag_viz, peso in self._cooccurrence_map.get(tag, []):
                if tag_viz in tags_conhecidas:
                    boost_por_tag[tag_viz] = boost_por_tag.get(tag_viz, 0.0) + peso

        if not boost_por_tag:
            return np.zeros(len(self._posts), dtype=np.float32)

        scores = np.zeros(len(self._posts), dtype=np.float32)
        for i, tags_post in enumerate(self._posts["tags_fitness"]):
            for tag_post in tags_post:
                scores[i] += boost_por_tag.get(tag_post, 0.0)

        max_score = scores.max()
        if max_score > 0:
            scores /= max_score
        return scores

    def _score_social(self) -> np.ndarray:
        if self._social_scores is not None and len(self._social_scores) == len(self._posts):
            return self._social_scores
        return np.zeros(len(self._posts), dtype=np.float32)

    def _score_popularidade(self) -> np.ndarray:
        """
        Retorna o vetor de popularidade pré-computado (normalizado em [0,1]).
        Se o artefato não foi carregado, retorna zeros.
        """
        if self._popularidade is not None and len(self._popularidade) == len(self._posts):
            return self._popularidade
        return np.zeros(len(self._posts), dtype=np.float32)

    def _score_time_decay(self, timestamp_entrada: int) -> np.ndarray:
        timestamps_posts = self._posts["creation_date"].values.astype(np.float64)
        delta_dias = np.abs(timestamps_posts - float(timestamp_entrada)) / MS_POR_DIA
        return np.exp(-LAMBDA_DECAY * delta_dias).astype(np.float32)

    def _score_user_affinity(self, user_id: int | None) -> np.ndarray:
        if user_id is None or self._user_tag_profile is None or self._user_tag_profile.empty:
            return np.zeros(len(self._posts), dtype=np.float32)

        perfil_user = self._user_tag_profile[self._user_tag_profile["user_id"] == int(user_id)]
        if perfil_user.empty:
            return np.zeros(len(self._posts), dtype=np.float32)

        tag_score = dict(zip(perfil_user["tag_name"], perfil_user["user_tag_affinity"]))
        scores = np.zeros(len(self._posts), dtype=np.float32)

        for i, tags_post in enumerate(self._posts["tags_fitness"]):
            if not tags_post:
                continue
            total = sum(float(tag_score.get(tag, 0.0)) for tag in tags_post)
            scores[i] = total / len(tags_post)

        max_score = scores.max()
        if max_score > 0:
            scores /= max_score
        return scores

    def _tem_perfil_usuario(self, user_id: int | None) -> bool:
        if user_id is None or self._user_tag_profile is None or self._user_tag_profile.empty:
            return False
        return bool((self._user_tag_profile["user_id"] == int(user_id)).any())

    def recomendar(
        self,
        tags: list[str],
        timestamp: int,
        top_k: int = 10,
        excluir_tags_exatas: bool = True,
        peso_popularidade: float = PESO_POPULARIDADE,
        user_id: int | None = None,
    ) -> pd.DataFrame:
        if self._posts is None:
            raise RuntimeError("Modelo não carregado. Chame .carregar() primeiro.")

        tags_norm = [t.strip() for t in tags if t.strip()]
        if not tags_norm:
            raise ValueError("Lista de tags não pode ser vazia.")
        if peso_popularidade < 0:
            raise ValueError("peso_popularidade deve ser maior ou igual a zero.")

        sc = self._score_cosine(tags_norm)
        si = self._score_cooccurrence(tags_norm)
        st = self._score_time_decay(timestamp)
        ss = self._score_social()
        usar_personalizacao = self._tem_perfil_usuario(user_id)

        if usar_personalizacao:
            su = self._score_user_affinity(user_id)
            score_final = (
                PESO_COSINE_PERSONALIZADO * sc
                + PESO_COOC_PERSONALIZADO * si
                + PESO_TIME * st
                + PESO_SOCIAL * ss
                + PESO_USER_AFFINITY * su
            )
        else:
            sp = self._score_popularidade()
            peso_total = PESO_COSINE + PESO_COOC + PESO_TIME + PESO_SOCIAL + peso_popularidade
            score_final = (
                PESO_COSINE * sc
                + PESO_COOC * si
                + PESO_TIME * st
                + PESO_SOCIAL * ss
                + peso_popularidade * sp
            ) / peso_total

        score_final = np.clip(score_final, 0.0, 1.0).astype(np.float32)

        resultado = self._posts.copy()
        resultado["relevance_score"] = score_final.round(4)

        if excluir_tags_exatas:
            tags_set = set(tags_norm)
            resultado = resultado[resultado["tags_fitness"].apply(lambda t: set(t) != tags_set)]

        resultado = resultado.sort_values("relevance_score", ascending=False).head(top_k)

        colunas_saida = [
            "message_type",
            "creation_date_iso",
            "tags_fitness",
            "content_length",
            "language",
            "relevance_score",
        ]
        return resultado[[c for c in colunas_saida if c in resultado.columns]].reset_index(drop=True)


_modelo: ModeloRecomendacao | None = None


def _get_modelo() -> ModeloRecomendacao:
    global _modelo
    if _modelo is None:
        _modelo = ModeloRecomendacao().carregar()
    return _modelo


def recomendar(
    tags: list[str],
    timestamp: int,
    top_k: int = 10,
    excluir_tags_exatas: bool = True,
    peso_popularidade: float = PESO_POPULARIDADE,
    user_id: int | None = None,
) -> pd.DataFrame:
    """
    Função de alto nível para recomendação de posts fitness.

    Parâmetros
    ----------
    tags : list[str]
        Nomes das tags de entrada (ex: ["Born_to_Run", "Superunknown"]).
    timestamp : int
        Timestamp em milissegundos do post de referência.
    top_k : int
        Quantos posts recomendar (padrão: 10).
    excluir_tags_exatas : bool
        Remove posts com conjunto de tags idêntico ao da entrada.
    peso_popularidade : float
        Peso do sinal de popularidade no score padrão/fallback (padrão: 0.10).
    user_id : int | None
        Identificador do usuário alvo; se houver perfil disponível, ativa personalização.

    Retorna
    -------
    pd.DataFrame com os top_k posts recomendados e seus scores de relevância.
    """
    return _get_modelo().recomendar(
        tags,
        timestamp,
        top_k,
        excluir_tags_exatas,
        peso_popularidade,
        user_id,
    )


def _cli() -> None:
    parser = argparse.ArgumentParser(
        description="Recomendação de posts fitness por tags e timestamp.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemplos:
  python treinamento/recomendar.py --tags "Born_to_Run,Superunknown" --timestamp 1320000000000
  python treinamento/recomendar.py --tags "Running_Free" --timestamp 1300000000000 --top-k 5 --user-id 123
  python treinamento/recomendar.py --listar-tags
        """,
    )
    parser.add_argument("--tags", type=str, help='Tags separadas por vírgula (ex: "Born_to_Run,Superunknown")')
    parser.add_argument("--timestamp", type=int, help="Timestamp em milissegundos do post de referência")
    parser.add_argument("--top-k", type=int, default=10, help="Número de recomendações (padrão: 10)")
    parser.add_argument(
        "--peso-popularidade",
        type=float,
        default=PESO_POPULARIDADE,
        help=f"Peso do sinal de popularidade no score padrão/fallback (padrão: {PESO_POPULARIDADE})",
    )
    parser.add_argument("--user-id", type=int, default=None, help="User ID para recomendação personalizada")
    parser.add_argument("--listar-tags", action="store_true", help="Lista todas as tags conhecidas pelo modelo")
    parser.add_argument(
        "--incluir-exatas",
        action="store_true",
        help="Inclui posts com conjunto de tags idêntico ao de entrada",
    )

    args = parser.parse_args()
    modelo = _get_modelo()

    if args.listar_tags:
        tags_conhecidas = sorted(modelo._vectorizer.classes_)
        print(f"Tags conhecidas pelo modelo ({len(tags_conhecidas)}):")
        for tag in tags_conhecidas:
            print(f"  {tag}")
        return

    if not args.tags or args.timestamp is None:
        parser.error("--tags e --timestamp são obrigatórios (ou use --listar-tags)")

    tags_entrada = [t.strip() for t in args.tags.split(",") if t.strip()]

    print("\nBuscando recomendações para:")
    print(f"  Tags      : {tags_entrada}")
    print(f"  Timestamp : {args.timestamp}")
    print(f"  User ID   : {args.user_id if args.user_id is not None else '(não informado)'}")
    print(f"  Top-K     : {args.top_k}")
    print(f"  Peso pop. : {args.peso_popularidade}")
    print()

    df = modelo.recomendar(
        tags=tags_entrada,
        timestamp=args.timestamp,
        top_k=args.top_k,
        excluir_tags_exatas=not args.incluir_exatas,
        peso_popularidade=args.peso_popularidade,
        user_id=args.user_id,
    )

    if df.empty:
        print("Nenhuma recomendação encontrada.")
        return

    print(f"=== Top {len(df)} recomendações ===\n")
    pd.set_option("display.max_colwidth", 60)
    pd.set_option("display.width", 120)
    print(df.to_string(index=True))


if __name__ == "__main__":
    _cli()
