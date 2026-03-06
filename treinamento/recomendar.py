"""
Módulo de recomendação de posts fitness.

Carrega os artefatos treinados e expõe a função principal:

    recomendar(tags, timestamp, top_k=10) -> pd.DataFrame

O score de relevância combina quatro sinais:
  - Similaridade de conteúdo  (0.40): coseno entre o vetor de tags de entrada e cada post
  - Co-ocorrência de tags     (0.25): boost para posts que contêm tags relacionadas às de entrada
  - Recência relativa         (0.15): decaimento exponencial pela distância temporal em dias
  - Influência social         (0.20): soma dos graus dos usuários que interagiram com o post

Entradas:
    tags      : List[str]  — nomes das tags (valores, não IDs)
    timestamp : int        — timestamp em milissegundos do post de referência
    top_k     : int        — número de recomendações

Saída:
    DataFrame com colunas:
        message_type, creation_date_iso, tags_fitness,
        content_length, language, relevance_score

Uso como script (CLI):
    python treinamento/recomendar.py --tags "Born_to_Run,Superunknown" --timestamp 1320000000000
    python treinamento/recomendar.py --tags "Running_Free" --timestamp 1300000000000 --top-k 5
"""

from __future__ import annotations

import argparse
import ast
import json
import math
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

ROOT = Path(__file__).resolve().parent.parent
DADOS_DIR = ROOT / "treinamento" / "dados"
MODELO_DIR = ROOT / "treinamento" / "modelo"
PESOS_OTIMOS_PATH = MODELO_DIR / "pesos_otimos.json"

# Pesos padrão do score híbrido
PESO_COSINE_PADRAO = 0.40
PESO_COOC_PADRAO = 0.25
PESO_TIME_PADRAO = 0.15
PESO_SOCIAL_PADRAO = 0.20

# Lambda do decaimento temporal (por dia)
LAMBDA_DECAY = 0.01
MS_POR_DIA = 86_400_000


def _carregar_pesos_otimos() -> tuple[float, float, float, float]:
    """Carrega pesos otimizados quando disponíveis; caso contrário usa padrão."""
    default = (PESO_COSINE_PADRAO, PESO_COOC_PADRAO, PESO_TIME_PADRAO, PESO_SOCIAL_PADRAO)

    if not PESOS_OTIMOS_PATH.exists():
        return default

    try:
        with open(PESOS_OTIMOS_PATH, "r", encoding="utf-8") as f:
            payload = json.load(f)

        pesos = (
            float(payload["w_cos"]),
            float(payload["w_cooc"]),
            float(payload["w_time"]),
            float(payload["w_social"]),
        )

        if any(w < 0 for w in pesos):
            return default
        if not math.isclose(sum(pesos), 1.0, rel_tol=0, abs_tol=1e-6):
            return default
        return pesos
    except Exception:
        return default


PESO_COSINE, PESO_COOC, PESO_TIME, PESO_SOCIAL = _carregar_pesos_otimos()


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

        # Social scores são opcionais — fallback para zeros se artefato ausente
        social_path = MODELO_DIR / "social_scores.npy"
        if social_path.exists():
            self._social_scores = np.load(social_path)
        else:
            self._social_scores = None

        # posts_cache.parquet garante alinhamento exato com post_matrix
        # (treino em split usa subconjunto de posts; posts_metadata tem todos)
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

    # ------------------------------------------------------------------
    # Sinais individuais
    # ------------------------------------------------------------------

    def _score_cosine(self, tags_entrada: list[str]) -> np.ndarray:
        """Similaridade coseno entre a entrada e cada post."""
        vetor_entrada = self._vectorizer.transform([tags_entrada]).astype(np.float32)
        if vetor_entrada.sum() == 0:
            return np.zeros(len(self._posts), dtype=np.float32)
        sims = cosine_similarity(vetor_entrada, self._post_matrix).flatten()
        return sims.astype(np.float32)

    def _score_cooccurrence(self, tags_entrada: list[str]) -> np.ndarray:
        """
        Boost baseado em tags co-ocorrentes.
        Para cada tag de entrada, obtém suas tags relacionadas com pesos.
        Soma os pesos das tags relacionadas presentes em cada post.
        """
        tags_conhecidas = set(self._vectorizer.classes_)
        boost_por_tag: dict[str, float] = {}

        for tag in tags_entrada:
            vizinhos = self._cooccurrence_map.get(tag, [])
            for tag_viz, peso in vizinhos:
                if tag_viz in tags_conhecidas:
                    boost_por_tag[tag_viz] = boost_por_tag.get(tag_viz, 0.0) + peso

        if not boost_por_tag:
            return np.zeros(len(self._posts), dtype=np.float32)

        scores = np.zeros(len(self._posts), dtype=np.float32)
        for i, tags_post in enumerate(self._posts["tags_fitness"]):
            for tag_post in tags_post:
                scores[i] += boost_por_tag.get(tag_post, 0.0)

        # Normaliza para [0, 1]
        max_score = scores.max()
        if max_score > 0:
            scores /= max_score

        return scores

    def _score_social(self) -> np.ndarray:
        """
        Retorna o vetor de influência social pré-computado (já normalizado em [0,1]).
        Se o artefato não foi carregado, retorna zeros.
        """
        if self._social_scores is not None and len(self._social_scores) == len(self._posts):
            return self._social_scores
        return np.zeros(len(self._posts), dtype=np.float32)

    def _score_time_decay(self, timestamp_entrada: int) -> np.ndarray:
        """Decaimento exponencial pela distância temporal em dias."""
        timestamps_posts = self._posts["creation_date"].values.astype(np.float64)
        delta_dias = np.abs(timestamps_posts - float(timestamp_entrada)) / MS_POR_DIA
        decaimento = np.exp(-LAMBDA_DECAY * delta_dias).astype(np.float32)
        return decaimento

    # ------------------------------------------------------------------
    # Recomendação principal
    # ------------------------------------------------------------------

    def recomendar(
        self,
        tags: list[str],
        timestamp: int,
        top_k: int = 10,
        excluir_tags_exatas: bool = True,
    ) -> pd.DataFrame:
        """
        Retorna os top_k posts mais relevantes dado um conjunto de tags e timestamp.

        Parâmetros
        ----------
        tags : list[str]
            Nomes das tags do post de referência.
        timestamp : int
            Timestamp em milissegundos do post de referência.
        top_k : int
            Número de recomendações a retornar.
        excluir_tags_exatas : bool
            Se True, remove posts que possuem exatamente o mesmo conjunto de tags
            que a entrada (evita recomendar o próprio post).

        Retorna
        -------
        DataFrame com colunas:
            message_type, creation_date_iso, tags_fitness,
            content_length, language, relevance_score
        """
        if self._posts is None:
            raise RuntimeError("Modelo não carregado. Chame .carregar() primeiro.")

        tags_norm = [t.strip() for t in tags if t.strip()]
        if not tags_norm:
            raise ValueError("Lista de tags não pode ser vazia.")

        sc = self._score_cosine(tags_norm)
        si = self._score_cooccurrence(tags_norm)
        st = self._score_time_decay(timestamp)
        ss = self._score_social()

        score_final = PESO_COSINE * sc + PESO_COOC * si + PESO_TIME * st + PESO_SOCIAL * ss

        resultado = self._posts.copy()
        resultado["relevance_score"] = score_final.round(4)

        if excluir_tags_exatas:
            tags_set = set(tags_norm)
            mascara = resultado["tags_fitness"].apply(lambda t: set(t) != tags_set)
            resultado = resultado[mascara]

        resultado = resultado.sort_values("relevance_score", ascending=False).head(top_k)

        colunas_saida = [
            "message_type",
            "creation_date_iso",
            "tags_fitness",
            "content_length",
            "language",
            "relevance_score",
        ]
        colunas_disponiveis = [c for c in colunas_saida if c in resultado.columns]
        return resultado[colunas_disponiveis].reset_index(drop=True)


# ------------------------------------------------------------------
# Instância global (lazy-loaded)
# ------------------------------------------------------------------

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

    Retorna
    -------
    pd.DataFrame com os top_k posts recomendados e seus scores de relevância.

    Exemplo
    -------
    >>> from treinamento.recomendar import recomendar
    >>> df = recomendar(["Born_to_Run"], timestamp=1320000000000, top_k=5)
    >>> print(df)
    """
    return _get_modelo().recomendar(tags, timestamp, top_k, excluir_tags_exatas)


# ------------------------------------------------------------------
# CLI
# ------------------------------------------------------------------

def _cli() -> None:
    parser = argparse.ArgumentParser(
        description="Recomendação de posts fitness por tags e timestamp.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemplos:
  python treinamento/recomendar.py --tags "Born_to_Run,Superunknown" --timestamp 1320000000000
  python treinamento/recomendar.py --tags "Running_Free" --timestamp 1300000000000 --top-k 5
  python treinamento/recomendar.py --listar-tags
        """,
    )
    parser.add_argument(
        "--tags",
        type=str,
        help='Tags separadas por vírgula (ex: "Born_to_Run,Superunknown")',
    )
    parser.add_argument(
        "--timestamp",
        type=int,
        help="Timestamp em milissegundos do post de referência",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=10,
        help="Número de recomendações (padrão: 10)",
    )
    parser.add_argument(
        "--listar-tags",
        action="store_true",
        help="Lista todas as tags conhecidas pelo modelo",
    )
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

    print(f"\nBuscando recomendações para:")
    print(f"  Tags      : {tags_entrada}")
    print(f"  Timestamp : {args.timestamp}")
    print(f"  Top-K     : {args.top_k}")
    print()

    df = modelo.recomendar(
        tags=tags_entrada,
        timestamp=args.timestamp,
        top_k=args.top_k,
        excluir_tags_exatas=not args.incluir_exatas,
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
