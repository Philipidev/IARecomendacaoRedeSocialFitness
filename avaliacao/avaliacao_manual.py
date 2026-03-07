"""Executor de avaliação manual reproduzível para o recomendador.

Uso:
    python avaliacao/avaliacao_manual.py
    python avaliacao/avaliacao_manual.py --casos avaliacao/casos_manuais.yaml --saida avaliacao/resultados/avaliacao_manual.md
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
import sys

import pandas as pd
import json

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from progress_utils import IterationProgress
from treinamento.recomendar import recomendar

MS_POR_DIA = 86_400_000


@dataclass
class ResultadoCriterio:
    nome: str
    esperado: str
    observado: str
    aprovado: bool


def _fmt_bool(value: bool) -> str:
    return "✅" if value else "❌"


def _carregar_casos(caminho: Path) -> list[dict[str, Any]]:
    with caminho.open("r", encoding="utf-8") as f:
        conteudo = f.read()

    try:
        import yaml  # type: ignore

        payload = yaml.safe_load(conteudo) or {}
    except ModuleNotFoundError:
        payload = json.loads(conteudo)
    casos = payload.get("casos", [])
    if not isinstance(casos, list) or not casos:
        raise ValueError("Arquivo de casos inválido: chave 'casos' ausente ou vazia.")
    return casos


def _tags_set(tags: Any) -> set[str]:
    if isinstance(tags, list):
        return {str(t) for t in tags}
    return set()


def _avaliar_criterios(caso: dict[str, Any], saida: pd.DataFrame, tags_entrada: list[str], timestamp: int) -> list[ResultadoCriterio]:
    criterios = caso.get("criterios_saida", {})
    resultados: list[ResultadoCriterio] = []

    if saida.empty:
        for nome, dados in criterios.items():
            resultados.append(
                ResultadoCriterio(
                    nome=nome,
                    esperado=dados.get("esperado", ""),
                    observado="Sem resultados para avaliar.",
                    aprovado=False,
                )
            )
        return resultados

    total = len(saida)
    tags_entrada_set = set(tags_entrada)
    tags_series = saida["tags_fitness"].apply(_tags_set)

    # 1) aderência temática
    c1 = criterios.get("aderencia_tematica", {})
    min_prop = float(c1.get("proporcao_minima", 0.0))
    prop_aderencia = float((tags_series.apply(lambda s: bool(s & tags_entrada_set))).mean())
    resultados.append(
        ResultadoCriterio(
            nome="aderencia_tematica",
            esperado=c1.get("esperado", ""),
            observado=f"{prop_aderencia:.2%} dos itens possuem interseção direta com as tags de entrada (mínimo {min_prop:.0%}).",
            aprovado=prop_aderencia >= min_prop,
        )
    )

    # 2) recência
    c2 = criterios.get("recencia", {})
    max_delta = int(c2.get("max_delta_dias", 10**9))
    min_prop_rec = float(c2.get("proporcao_minima", 0.0))
    deltas = (saida["creation_timestamp_ms"] - timestamp).abs() / MS_POR_DIA
    prop_recente = float((deltas <= max_delta).mean())
    mediana_delta = float(deltas.median())
    resultados.append(
        ResultadoCriterio(
            nome="recencia",
            esperado=c2.get("esperado", ""),
            observado=(
                f"{prop_recente:.2%} dos itens dentro de {max_delta} dias (mínimo {min_prop_rec:.0%}); "
                f"mediana de distância temporal: {mediana_delta:.1f} dias."
            ),
            aprovado=prop_recente >= min_prop_rec,
        )
    )

    # 3) variedade
    c3 = criterios.get("variedade", {})
    min_unicos = int(c3.get("minimo_conjuntos_unicos", 1))
    unicos = len({tuple(sorted(s)) for s in tags_series})
    resultados.append(
        ResultadoCriterio(
            nome="variedade",
            esperado=c3.get("esperado", ""),
            observado=f"{unicos} conjuntos únicos de tags no top-{total} (mínimo {min_unicos}).",
            aprovado=unicos >= min_unicos,
        )
    )

    # 4) ausência de irrelevantes
    c4 = criterios.get("ausencia_itens_irrelevantes", {})
    bloqueadas = set(c4.get("tags_bloqueadas", []))
    min_prop_clean = float(c4.get("proporcao_minima", 0.0))
    sem_irrelevantes = float((tags_series.apply(lambda s: len(s & bloqueadas) == 0)).mean())
    resultados.append(
        ResultadoCriterio(
            nome="ausencia_itens_irrelevantes",
            esperado=c4.get("esperado", ""),
            observado=(
                f"{sem_irrelevantes:.2%} dos itens sem tags bloqueadas {sorted(bloqueadas)} "
                f"(mínimo {min_prop_clean:.0%})."
            ),
            aprovado=sem_irrelevantes >= min_prop_clean,
        )
    )

    return resultados


def _preparar_saida(df: pd.DataFrame) -> pd.DataFrame:
    saida = df.copy()
    if "creation_date_iso" in saida.columns:
        dt = pd.to_datetime(saida["creation_date_iso"], errors="coerce", utc=True)
        saida["creation_timestamp_ms"] = (dt.view("int64") // 10**6).astype("Int64")
    elif "creation_date" in saida.columns:
        saida["creation_timestamp_ms"] = pd.to_numeric(saida["creation_date"], errors="coerce").astype("Int64")
    else:
        saida["creation_timestamp_ms"] = pd.Series([pd.NA] * len(saida), dtype="Int64")

    saida["tags_fitness"] = saida.get("tags_fitness", pd.Series([[] for _ in range(len(saida))])).apply(
        lambda x: x if isinstance(x, list) else []
    )
    return saida


def _tabela_markdown(df: pd.DataFrame, colunas: list[str]) -> str:
    if df.empty:
        return "Sem resultados retornados.\n"
    return df[colunas].to_markdown(index=True)


def _gerar_relatorio(casos_resultados: list[dict[str, Any]], destino: Path) -> None:
    destino.parent.mkdir(parents=True, exist_ok=True)

    linhas: list[str] = []
    linhas.append("# Avaliação Manual do Sistema de Recomendação\n")
    linhas.append(f"_Gerado em {datetime.now(timezone.utc).isoformat()} (UTC)._\n")

    total_casos = len(casos_resultados)
    aprovados = sum(1 for c in casos_resultados if c["aprovado_global"])
    linhas.append("## Resumo Geral\n")
    linhas.append(f"- Casos executados: **{total_casos}**")
    linhas.append(f"- Casos aprovados (todos os critérios): **{aprovados}/{total_casos}**\n")

    for caso in casos_resultados:
        linhas.append(f"## {caso['id']} — {caso['titulo']}\n")
        linhas.append(f"**Descrição:** {caso['descricao']}  ")
        linhas.append(f"**Entrada:** tags={caso['entrada']['tags']}, timestamp={caso['entrada']['timestamp']}, top_k={caso['entrada']['top_k']}\n")

        if caso.get("erro"):
            linhas.append(f"**Erro na execução:** `{caso['erro']}`\n")
        else:
            linhas.append("### Top-K retornado\n")
            preview_cols = [
                c
                for c in ["message_type", "creation_date_iso", "tags_fitness", "language", "relevance_score"]
                if c in caso["saida"].columns
            ]
            linhas.append(_tabela_markdown(caso["saida"], preview_cols))
            linhas.append("")

        linhas.append("### Avaliação qualitativa\n")
        linhas.append("| Critério | Esperado | Observado | Status |")
        linhas.append("|---|---|---|---|")
        for crit in caso["criterios_avaliados"]:
            linhas.append(
                f"| {crit.nome} | {crit.esperado} | {crit.observado} | {_fmt_bool(crit.aprovado)} |"
            )

        linhas.append("")

    linhas.append("## Conclusão\n")
    if aprovados == total_casos:
        linhas.append("Todos os cenários passaram nos critérios qualitativos definidos.")
    elif aprovados == 0:
        linhas.append(
            "Nenhum cenário foi integralmente aprovado. Revise os pesos do modelo, o dataset de treino "
            "ou ajuste os critérios esperados para refletir o comportamento desejado."
        )
    else:
        linhas.append(
            "Parte dos cenários foi aprovada. Recomenda-se investigar os critérios reprovados e repetir "
            "a avaliação após novos ajustes no pipeline/modelo."
        )

    destino.write_text("\n".join(linhas) + "\n", encoding="utf-8")


def _gerar_sumario_json(casos_resultados: list[dict[str, Any]]) -> dict[str, Any]:
    total_casos = len(casos_resultados)
    aprovados = sum(1 for c in casos_resultados if c["aprovado_global"])
    criterios_total = sum(len(c["criterios_avaliados"]) for c in casos_resultados)
    criterios_aprovados = sum(
        1
        for c in casos_resultados
        for criterio in c["criterios_avaliados"]
        if criterio.aprovado
    )
    return {
        "total_casos": total_casos,
        "casos_aprovados": aprovados,
        "taxa_aprovacao_casos": (aprovados / total_casos) if total_casos else 0.0,
        "total_criterios": criterios_total,
        "criterios_aprovados": criterios_aprovados,
        "taxa_aprovacao_criterios": (
            criterios_aprovados / criterios_total if criterios_total else 0.0
        ),
    }


def executar_avaliacao(
    casos_path: Path,
    saida_path: Path,
    model_dir: str | None = None,
    saida_json: Path | None = None,
) -> dict[str, Any]:
    casos = _carregar_casos(casos_path)
    resultados: list[dict[str, Any]] = []
    progress = IterationProgress(
        total=len(casos),
        label="Avaliação manual",
        every_percent=10,
    )
    if casos:
        progress.start("Executando casos")

    for idx, caso in enumerate(casos, start=1):
        entrada = caso.get("entrada", {})
        tags = entrada.get("tags", [])
        timestamp = int(entrada.get("timestamp", 0))
        top_k = int(entrada.get("top_k", 10))

        registro: dict[str, Any] = {
            "id": caso.get("id", "sem_id"),
            "titulo": caso.get("titulo", "Sem título"),
            "descricao": caso.get("descricao", ""),
            "entrada": {
                "tags": tags,
                "timestamp": timestamp,
                "top_k": top_k,
            },
            "saida": pd.DataFrame(),
            "criterios_avaliados": [],
            "aprovado_global": False,
            "erro": None,
        }

        try:
            df = recomendar(
                tags=tags,
                timestamp=timestamp,
                top_k=top_k,
                model_dir=model_dir,
            )
            saida = _preparar_saida(df)
            criterios = _avaliar_criterios(caso, saida, tags, timestamp)
            registro["saida"] = saida
            registro["criterios_avaliados"] = criterios
            registro["aprovado_global"] = all(c.aprovado for c in criterios) if criterios else False
        except Exception as exc:  # noqa: BLE001
            criterios = _avaliar_criterios(caso, pd.DataFrame(), tags, timestamp)
            registro["criterios_avaliados"] = criterios
            registro["erro"] = str(exc)
            registro["aprovado_global"] = False

        resultados.append(registro)
        if casos:
            progress.log(idx, detail=f"Caso atual: {registro['id']}")

    _gerar_relatorio(resultados, saida_path)
    if casos:
        progress.finish("Casos executados")
    resumo = _gerar_sumario_json(resultados)
    if saida_json is not None:
        saida_json.parent.mkdir(parents=True, exist_ok=True)
        saida_json.write_text(json.dumps(resumo, ensure_ascii=False, indent=2), encoding="utf-8")
    return resumo


def main() -> None:
    parser = argparse.ArgumentParser(description="Executa avaliação manual reproduzível do recomendador")
    parser.add_argument("--casos", default="avaliacao/casos_manuais.yaml", help="Arquivo YAML com casos")
    parser.add_argument(
        "--saida",
        default="avaliacao/resultados/avaliacao_manual.md",
        help="Arquivo markdown de saída",
    )
    parser.add_argument(
        "--saida-json",
        default=None,
        help="Arquivo JSON opcional com resumo consolidado da avaliação manual",
    )
    parser.add_argument(
        "--model-dir",
        default=None,
        help="Diretório do modelo/ranker a ser avaliado",
    )
    args = parser.parse_args()

    executar_avaliacao(
        Path(args.casos),
        Path(args.saida),
        model_dir=args.model_dir,
        saida_json=Path(args.saida_json) if args.saida_json else None,
    )
    print(f"Relatório gerado em: {args.saida}")


if __name__ == "__main__":
    main()
