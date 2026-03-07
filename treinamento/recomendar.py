"""
Módulo de recomendação de posts fitness.

Expõe a função principal `recomendar()` e o carregamento plugável de rankers por
`model_dir`, suportando tanto o baseline híbrido quanto o ranker LTR.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from treinamento.model_utils import DEFAULT_MODEL_DIR, resolve_model_dir
from treinamento.rankers import (
    PESO_POPULARIDADE,
    WeightedHybridRanker,
    load_ranker,
)

DADOS_DIR = ROOT / "treinamento" / "dados"
MODELO_DIR = DEFAULT_MODEL_DIR
PESOS_OTIMOS_PATH = MODELO_DIR / "pesos_otimos.json"


class ModeloRecomendacao(WeightedHybridRanker):
    """Façade compatível com o baseline híbrido legado."""


_modelos_cache: dict[str, object] = {}


def _get_modelo(model_dir: str | Path | None = None):
    resolved = resolve_model_dir(model_dir)
    key = str(resolved.resolve())
    if key not in _modelos_cache:
        _modelos_cache[key] = load_ranker(resolved)
    return _modelos_cache[key]


def recomendar(
    tags: list[str],
    timestamp: int,
    top_k: int = 10,
    excluir_tags_exatas: bool = True,
    peso_popularidade: float = PESO_POPULARIDADE,
    user_id: int | None = None,
    model_dir: str | Path | None = None,
) -> pd.DataFrame:
    return _get_modelo(model_dir).recommend_df(
        tags=tags,
        timestamp=timestamp,
        top_k=top_k,
        excluir_tags_exatas=excluir_tags_exatas,
        peso_popularidade=peso_popularidade,
        user_id=user_id,
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
    parser.add_argument(
        "--model-dir",
        type=str,
        default=str(MODELO_DIR),
        help="Diretório do modelo a carregar",
    )

    args = parser.parse_args()
    modelo = _get_modelo(args.model_dir)

    if args.listar_tags:
        vectorizer = None
        if hasattr(modelo, "_vectorizer"):
            vectorizer = modelo._vectorizer
        elif getattr(modelo, "artifacts", None) is not None:
            vectorizer = modelo.artifacts.vectorizer

        tags_conhecidas = sorted(vectorizer.classes_) if vectorizer is not None else []
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
    print(f"  Model dir : {resolve_model_dir(args.model_dir)}")
    print()

    df = modelo.recommend_df(
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
