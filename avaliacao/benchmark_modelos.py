from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dataset_context import dataset_context, manifest_path
from pipeline_contracts import (
    normalize_split_config,
    split_signature,
    split_signature_from_manifest_file,
    split_signature_from_metadata,
)
from progress_utils import IterationProgress, StageProgress
from treinamento.model_utils import MODELOS_DIR, merge_model_metadata, now_iso, rel_path

PREPARAR_SCRIPT = ROOT / "treinamento" / "preparacao_dados.py"
DIVIDIR_SCRIPT = ROOT / "treinamento" / "dividir_dataset.py"
TREINAR_SCRIPT = ROOT / "treinamento" / "treinar.py"
PREPARAR_LTR_SCRIPT = ROOT / "treinamento" / "preparar_dataset_ltr.py"
TREINAR_LTR_SCRIPT = ROOT / "treinamento" / "treinar_ltr.py"
AVALIAR_SCRIPT = ROOT / "avaliacao" / "avaliar_modelo.py"
AVALIACAO_MANUAL_SCRIPT = ROOT / "avaliacao" / "avaliacao_manual.py"
AVALIAR_POPULARIDADE_SCRIPT = ROOT / "avaliacao" / "avaliar_popularidade.py"
OTIMIZAR_PESOS_SCRIPT = ROOT / "avaliacao" / "otimizar_pesos.py"

DEFAULT_CONFIG = ROOT / "casos_uso_tcc.json"
DEFAULT_RESULTS_DIR = ROOT / "avaliacao" / "resultados"
METRIC_NDCG_10 = "ndcg@10"
METRIC_MRR_10 = "mrr@10"


def _resolve_path(path_str: str | None, default_path: Path) -> Path:
    if not path_str:
        return default_path
    path = Path(path_str)
    return path if path.is_absolute() else ROOT / path


def _run_python_script(script_path: Path, args: list[str] | None = None) -> None:
    args = args or []
    cmd = [sys.executable, str(script_path), *args]
    printable = " ".join([rel_path(script_path), *args])
    print(f"\n[Execução] python {printable}\n")
    subprocess.run(cmd, cwd=ROOT, check=True)


def _load_config(config_path: Path) -> dict[str, Any]:
    if not config_path.exists():
        raise FileNotFoundError(f"Arquivo de configuração não encontrado: {config_path}")
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("casos_uso_tcc.json deve conter um objeto JSON na raiz.")
    modelos = payload.get("modelos")
    if not isinstance(modelos, list) or not modelos:
        raise ValueError("casos_uso_tcc.json deve conter uma lista não vazia em 'modelos'.")
    return payload


def _split_signature(split_config: dict[str, Any]) -> tuple[float, float, float, int]:
    return (
        float(split_config.get("train", 0.70)),
        float(split_config.get("val", 0.15)),
        float(split_config.get("test", 0.15)),
        int(split_config.get("seed", 42)),
    )


def _resolve_runtime_root(
    override: str | None,
    runtime_default: Path,
    legacy_default: Path,
) -> Path:
    if not override:
        return runtime_default
    candidate = _resolve_path(override, runtime_default)
    try:
        if candidate.resolve() == legacy_default.resolve():
            return runtime_default
    except Exception:
        pass
    return candidate


def _artifact_size_mb(model_dir: Path) -> float:
    total_bytes = sum(
        path.stat().st_size
        for path in model_dir.rglob("*")
        if path.is_file()
    )
    return total_bytes / (1024 * 1024)


def _read_json_optional(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else {}


def _jsonable_records(df: pd.DataFrame) -> list[dict[str, Any]]:
    return json.loads(df.to_json(orient="records", force_ascii=False))


def _normalizar_avaliacoes(config_avaliacoes: dict[str, Any] | None) -> dict[str, bool]:
    base = {
        "offline": True,
        "manual": True,
        "popularidade": False,
        "otimizacao": False,
    }
    if isinstance(config_avaliacoes, dict):
        for key in base:
            if key in config_avaliacoes:
                base[key] = bool(config_avaliacoes[key])
    return base


def _features_enabled(model_cfg: dict[str, Any]) -> list[str]:
    dataset_ltr = model_cfg.get("dataset_ltr", {})
    features = dataset_ltr.get("features_enabled")
    if isinstance(features, list) and features:
        return [str(item) for item in features]
    if model_cfg.get("family") == "baseline_hibrido":
        return [
            "cosine_score",
            "cooccurrence_score",
            "time_decay_score",
            "social_score",
            "popularidade_score",
        ]
    return []


def _experiment_step_labels(
    model_cfg: dict[str, Any],
    *,
    split_needed: bool,
    manual_available: bool,
) -> list[str]:
    family = str(model_cfg["family"])
    params = model_cfg.get("params", {})
    avaliacoes = _normalizar_avaliacoes(model_cfg.get("avaliacoes"))

    steps: list[str] = []
    if split_needed:
        steps.append("Atualizando split train/val/test")

    steps.append("Treinando modelo base")

    if family == "baseline_hibrido" and params.get("usar_pesos_otimos"):
        steps.append("Otimizando pesos do baseline")
    if family == "ltr_lightgbm":
        steps.extend(
            [
                "Preparando dataset LTR",
                "Treinando ranker LTR",
            ]
        )
    if avaliacoes.get("offline", True):
        steps.append("Executando avaliação offline")
    if avaliacoes.get("manual", True) and manual_available:
        steps.append("Executando avaliação manual")
    if family == "baseline_hibrido" and avaliacoes.get("popularidade", False):
        steps.append("Executando avaliação de popularidade")

    steps.append("Consolidando métricas")
    return steps


def _write_baseline_weights(model_dir: Path, params: dict[str, Any]) -> Path:
    payload = {
        "w_cos": float(params.get("w_cos", 0.40)),
        "w_cooc": float(params.get("w_cooc", 0.25)),
        "w_time": float(params.get("w_time", 0.15)),
        "w_social": float(params.get("w_social", 0.20)),
        "metric_target": "manual_config",
        "metric_target_value": None,
        "top_k": None,
        "otimizacao": {
            "source": "casos_uso_tcc.json",
            "timestamp_utc": now_iso(),
        },
    }
    path = model_dir / "pesos_otimos.json"
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def _coletar_metricas(
    model_cfg: dict[str, Any],
    model_dir: Path,
    result_dir: Path,
    tempo_treinamento_s: float,
) -> dict[str, Any]:
    offline_json = _read_json_optional(result_dir / "offline" / "metricas_resumo.json")
    manual_json = _read_json_optional(result_dir / "manual" / "avaliacao_manual.json")
    popularidade_json = _read_json_optional(result_dir / "popularidade" / "metricas_antes_depois.json")
    metadata = _read_json_optional(model_dir / "metadata.json")

    row: dict[str, Any] = {
        "model_id": str(model_cfg["id"]),
        "family": str(model_cfg["family"]),
        "dataset_key": str(metadata.get("dataset", {}).get("dataset_key", "")),
        "descricao": str(model_cfg.get("descricao", "")),
        "metric_target": str(model_cfg.get("metric_target", METRIC_NDCG_10)),
        "split_seed": int(model_cfg.get("split_config", {}).get("seed", 42)),
        "split_config": json.dumps(model_cfg.get("split_config", {}), ensure_ascii=False),
        "feature_set": ", ".join(_features_enabled(model_cfg)),
        "params_resumidos": json.dumps(model_cfg.get("params", {}), ensure_ascii=False, sort_keys=True),
        "model_dir": rel_path(model_dir),
        "timestamp_execucao": now_iso(),
        "tempo_treinamento_s": tempo_treinamento_s,
        "latencia_p50_ms": 0.0,
        "latencia_p95_ms": 0.0,
        "artifact_size_mb": _artifact_size_mb(model_dir),
        "manual_taxa_aprovacao": float(manual_json.get("taxa_aprovacao_casos", 0.0)),
        "manual_taxa_criterios": float(manual_json.get("taxa_aprovacao_criterios", 0.0)),
        "popularidade_delta_ndcg@10": float(
            popularidade_json.get("delta", {}).get(METRIC_NDCG_10, 0.0)
        )
        if popularidade_json
        else 0.0,
        "model_split_signature": split_signature_from_metadata(metadata) or "",
    }

    if offline_json:
        ranking = offline_json.get("ranking_metrics", {})
        business = offline_json.get("business_metrics", {})
        meta = offline_json.get("metadata", {})
        row.update(ranking)
        row.update(
            {
                "catalog_coverage": float(business.get("catalog_coverage", 0.0)),
                "diversity": float(business.get("intra_list_diversity_tags", 0.0)),
                "novelty": float(business.get("novelty_inverse_popularity", 0.0)),
                "avg_recency_days": float(business.get("avg_recommended_recency_days", 0.0)),
                "latencia_p50_ms": float(meta.get("latencia_inferencia_ms_p50", 0.0)),
                "latencia_p95_ms": float(meta.get("latencia_inferencia_ms_p95", 0.0)),
                "offline_split_signature": str(meta.get("split_signature", "")),
                "offline_split_consistente": bool(meta.get("split_consistente", False)),
            }
        )

    if metadata.get("ltr", {}).get("best_iteration") is not None:
        row["ltr_best_iteration"] = int(metadata["ltr"]["best_iteration"])

    row["resultado_offline_stale"] = bool(
        row.get("offline_split_signature")
        and row.get("model_split_signature")
        and row["offline_split_signature"] != row["model_split_signature"]
    )
    metric_target = row["metric_target"]
    row["metric_target_value"] = float(row.get(metric_target, 0.0) or 0.0)
    return row


def _markdown_benchmark(df: pd.DataFrame, best_row: pd.Series) -> str:
    preview_cols = [
        col
        for col in [
            "model_id",
            "family",
            "metric_target",
            "metric_target_value",
            METRIC_NDCG_10,
            "map@10",
            "recall@10",
            "precision@10",
            "hitrate@10",
            METRIC_MRR_10,
            "resultado_offline_stale",
            "tempo_treinamento_s",
            "latencia_p95_ms",
        ]
        if col in df.columns
    ]
    table_md = df[preview_cols].to_markdown(index=False) if not df.empty else "Sem resultados."
    return "\n".join(
        [
            "# Benchmark de Modelos do TCC",
            "",
            f"Gerado em `{datetime.now(timezone.utc).isoformat()}`.",
            "",
            f"Melhor modelo: **{best_row['model_id']}** (`{best_row['metric_target']}` = {best_row['metric_target_value']:.4f})",
            "",
            "## Tabela resumida",
            "",
            table_md,
            "",
        ]
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Executa benchmark multi-modelo orientado ao TCC.")
    parser.add_argument("--config", type=str, default=str(DEFAULT_CONFIG))
    parser.add_argument("--results-dir", type=str, default=str(DEFAULT_RESULTS_DIR))
    parser.add_argument(
        "--dataset-key",
        type=str,
        default=None,
        help="Namespace lógico do dataset ativo; se omitido, usa layout legado",
    )
    parser.add_argument(
        "--dataset-path",
        type=str,
        default=None,
        help="Caminho opcional do dataset para registrar proveniência nos modelos",
    )
    parser.add_argument(
        "--scale-factor",
        type=str,
        default=None,
        help="Scale factor opcional para registrar proveniência nos modelos",
    )
    parser.add_argument(
        "--model-ids",
        nargs="+",
        default=None,
        help="Subconjunto opcional de model_id para executar no benchmark",
    )
    args = parser.parse_args()

    config_path = _resolve_path(args.config, DEFAULT_CONFIG)
    config = _load_config(config_path)
    selected_model_ids = [str(model_id) for model_id in (args.model_ids or [])]
    configured_model_ids = [str(cfg.get("id")) for cfg in config["modelos"] if isinstance(cfg, dict)]
    unknown_model_ids = [
        model_id for model_id in selected_model_ids if model_id not in configured_model_ids
    ]
    if unknown_model_ids:
        raise ValueError(
            "model_id(s) não encontrado(s) em casos_uso_tcc.json: "
            + ", ".join(unknown_model_ids)
        )

    benchmark_cfg = config.get("benchmark", {})
    runtime_context = dataset_context(
        dataset_key=args.dataset_key,
        dataset_path=args.dataset_path,
        scale_factor=args.scale_factor,
    )
    results_default = runtime_context.results_dir
    results_override = benchmark_cfg.get("resultados_dir")
    if results_override is None:
        if args.dataset_key and args.results_dir == str(DEFAULT_RESULTS_DIR):
            results_override = None
        else:
            results_override = args.results_dir
    results_dir = _resolve_runtime_root(
        results_override,
        results_default,
        DEFAULT_RESULTS_DIR,
    )
    modelos_root = _resolve_runtime_root(
        benchmark_cfg.get("modelos_dir"),
        runtime_context.models_dir,
        MODELOS_DIR,
    )
    casos_manuais = _resolve_path(
        benchmark_cfg.get("casos_manuais"),
        ROOT / "avaliacao" / "casos_manuais.yaml",
    )

    print("=== Benchmark multi-modelo do TCC ===")
    print(f"Dataset namespace: {runtime_context.dataset_key or 'legado'}")
    print(f"Config          : {config_path}")
    print(f"Resultados      : {results_dir}")
    print(f"Modelos         : {modelos_root}")
    enabled_models = [
        cfg
        for cfg in config["modelos"]
        if cfg.get("enabled", True)
        and (not selected_model_ids or str(cfg["id"]) in selected_model_ids)
    ]
    print(f"Modelos ativos  : {len(enabled_models)}")
    if selected_model_ids:
        print(f"Filtro model_id : {', '.join(selected_model_ids)}")

    if not enabled_models:
        raise RuntimeError(
            "Nenhum modelo habilitado correspondente foi encontrado no casos_uso_tcc.json."
        )
    if not casos_manuais.exists():
        print(f"Aviso           : casos manuais não encontrados em {casos_manuais}.")

    preparar_args: list[str] = []
    if args.dataset_key:
        preparar_args.extend(["--dataset-key", args.dataset_key])
    if args.dataset_path:
        preparar_args.extend(["--dataset-path", args.dataset_path])
    if args.scale_factor:
        preparar_args.extend(["--scale-factor", args.scale_factor])
    _run_python_script(PREPARAR_SCRIPT, preparar_args)

    last_split: tuple[float, float, float, int] | None = None
    rows: list[dict[str, Any]] = []
    benchmark_progress = IterationProgress(
        total=len(enabled_models),
        label="Benchmark modelos",
        every_percent=5,
    )
    benchmark_progress.start("Iniciando lote de experimentos")

    for model_index, model_cfg in enumerate(enabled_models, start=1):
        model_id = str(model_cfg["id"])
        family = str(model_cfg["family"])
        split_cfg = model_cfg.get("split_config", {})
        split_tuple = _split_signature(split_cfg)
        split_needed = last_split != split_tuple
        split_cfg_payload = normalize_split_config(split_cfg)
        split_signature_hash = split_signature(split_cfg_payload)
        exp_progress = StageProgress(
            total_stages=len(
                _experiment_step_labels(
                    model_cfg,
                    split_needed=split_needed,
                    manual_available=casos_manuais.exists(),
                )
            ),
            label=f"Experimento {model_id}",
        )

        benchmark_progress.log(
            current=model_index - 1,
            detail=f"Iniciando {model_id}",
            force=True,
        )
        print(f"\n=== Experimento: {model_id} ({family}) ===")

        if split_needed:
            exp_progress.step("Atualizando split train/val/test")
            _run_python_script(
                DIVIDIR_SCRIPT,
                [
                    *(["--dataset-key", args.dataset_key] if args.dataset_key else []),
                    *(["--dataset-path", args.dataset_path] if args.dataset_path else []),
                    *(["--scale-factor", args.scale_factor] if args.scale_factor else []),
                    "--train",
                    str(split_tuple[0]),
                    "--val",
                    str(split_tuple[1]),
                    "--test",
                    str(split_tuple[2]),
                    "--seed",
                    str(split_tuple[3]),
                ],
            )
            last_split = split_tuple

        model_dir = modelos_root / model_id
        result_dir = results_dir / "modelos" / model_id
        result_dir.mkdir(parents=True, exist_ok=True)

        train_started = perf_counter()

        train_args = [
            "--model-dir",
            str(model_dir),
            "--experiment-id",
            model_id,
        ]
        if args.dataset_key:
            train_args.extend(["--dataset-key", args.dataset_key])
        if args.dataset_path:
            train_args.extend(["--dataset-path", args.dataset_path])
        if args.scale_factor:
            train_args.extend(["--scale-factor", args.scale_factor])
        training_cfg = model_cfg.get("training", {})
        if training_cfg.get("dataset_completo"):
            train_args.append("--dataset-completo")
        elif training_cfg.get("catalogo_completo", True):
            train_args.append("--catalogo-completo")

        exp_progress.step("Treinando modelo base")
        _run_python_script(TREINAR_SCRIPT, train_args)

        params = model_cfg.get("params", {})
        avaliacoes = _normalizar_avaliacoes(model_cfg.get("avaliacoes"))

        merge_model_metadata(
            model_dir,
            {
                "id": model_id,
                "family": family,
                "descricao": model_cfg.get("descricao", ""),
                "enabled": bool(model_cfg.get("enabled", True)),
                "metric_target": model_cfg.get("metric_target", METRIC_NDCG_10),
                "top_k": model_cfg.get("top_k", [5, 10, 20]),
                "split_config": split_cfg_payload,
                "avaliacoes": avaliacoes,
                "notes": model_cfg.get("notes", ""),
                "params": params,
                "benchmark": {
                    "config_path": rel_path(config_path),
                    "result_dir": rel_path(result_dir),
                },
                "training": {
                    "split_config": split_cfg_payload,
                    "split_signature": split_signature_hash,
                },
            },
        )

        if family == "baseline_hibrido":
            if params.get("usar_pesos_otimos"):
                exp_progress.step("Otimizando pesos do baseline")
                _run_python_script(
                    OTIMIZAR_PESOS_SCRIPT,
                    [
                        "--model-dir",
                        str(model_dir),
                        "--grid-step",
                        str(params.get("grid_step", 0.1)),
                        "--random-search",
                        str(params.get("random_search", 0)),
                        "--top-k",
                        str(params.get("otimizacao_top_k", 10)),
                        "--max-queries",
                        str(params.get("max_queries_otimizacao", 300)),
                        "--seed",
                        str(split_tuple[3]),
                        "--out-csv",
                        str(result_dir / "pesos_experimentos.csv"),
                    ],
                )
            else:
                _write_baseline_weights(model_dir, params)

        elif family == "ltr_lightgbm":
            dataset_cfg = model_cfg.get("dataset_ltr", {})
            features_enabled = _features_enabled(model_cfg)
            exp_progress.step("Preparando dataset LTR")
            _run_python_script(
                PREPARAR_LTR_SCRIPT,
                [
                    "--model-dir",
                    str(model_dir),
                    "--train-out",
                    str(model_dir / "ltr_train.parquet"),
                    "--val-out",
                    str(model_dir / "ltr_val.parquet"),
                    "--meta-out",
                    str(model_dir / "ltr_dataset_meta.json"),
                    "--negatives-per-query",
                    str(dataset_cfg.get("negatives_per_query", 50)),
                    "--hard-negative-topn",
                    str(dataset_cfg.get("hard_negative_topn", 500)),
                    "--max-queries-train",
                    str(dataset_cfg.get("max_queries_train", 500)),
                    "--max-queries-val",
                    str(dataset_cfg.get("max_queries_val", 200)),
                    "--seed",
                    str(dataset_cfg.get("seed", split_tuple[3])),
                    "--features",
                    *features_enabled,
                ],
            )
            exp_progress.step("Treinando ranker LTR")
            _run_python_script(
                TREINAR_LTR_SCRIPT,
                [
                    "--model-dir",
                    str(model_dir),
                    "--train-dataset",
                    str(model_dir / "ltr_train.parquet"),
                    "--val-dataset",
                    str(model_dir / "ltr_val.parquet"),
                    "--meta-dataset",
                    str(model_dir / "ltr_dataset_meta.json"),
                    "--objective",
                    str(params.get("objective", "lambdarank")),
                    "--metric-at",
                    *[str(k) for k in params.get("metric_at", model_cfg.get("top_k", [5, 10, 20]))],
                    "--num-leaves",
                    str(params.get("num_leaves", 31)),
                    "--learning-rate",
                    str(params.get("learning_rate", 0.05)),
                    "--n-estimators",
                    str(params.get("n_estimators", 300)),
                    "--min-data-in-leaf",
                    str(params.get("min_data_in_leaf", 20)),
                    "--feature-fraction",
                    str(params.get("feature_fraction", 0.9)),
                    "--bagging-fraction",
                    str(params.get("bagging_fraction", 0.8)),
                    "--bagging-freq",
                    str(params.get("bagging_freq", 1)),
                    "--seed",
                    str(params.get("seed", split_tuple[3])),
                ],
            )
        else:
            raise ValueError(f"Família de modelo não suportada: {family}")

        tempo_treinamento_s = perf_counter() - train_started

        if avaliacoes.get("offline", True):
            exp_progress.step("Executando avaliação offline")
            _run_python_script(
                AVALIAR_SCRIPT,
                [
                    "--model-dir",
                    str(model_dir),
                    "--out-dir",
                    str(result_dir / "offline"),
                    "--k",
                    *[str(k) for k in model_cfg.get("top_k", [5, 10, 20])],
                ],
            )

        if avaliacoes.get("manual", True) and casos_manuais.exists():
            exp_progress.step("Executando avaliação manual")
            _run_python_script(
                AVALIACAO_MANUAL_SCRIPT,
                [
                    "--model-dir",
                    str(model_dir),
                    "--casos",
                    str(casos_manuais),
                    "--saida",
                    str(result_dir / "manual" / "avaliacao_manual.md"),
                    "--saida-json",
                    str(result_dir / "manual" / "avaliacao_manual.json"),
                ],
            )

        if family == "baseline_hibrido" and avaliacoes.get("popularidade", False):
            exp_progress.step("Executando avaliação de popularidade")
            _run_python_script(
                AVALIAR_POPULARIDADE_SCRIPT,
                [
                    "--model-dir",
                    str(model_dir),
                    "--k",
                    str(model_cfg.get("popularidade_k", 10)),
                    "--peso-depois",
                    str(params.get("peso_popularidade", 0.10)),
                    "--out-json",
                    str(result_dir / "popularidade" / "metricas_antes_depois.json"),
                ],
            )

        exp_progress.step("Consolidando métricas")
        row = _coletar_metricas(
            model_cfg=model_cfg,
            model_dir=model_dir,
            result_dir=result_dir,
            tempo_treinamento_s=tempo_treinamento_s,
        )
        rows.append(row)
        benchmark_progress.log(
            current=model_index,
            detail=f"Concluído {model_id}",
            force=True,
        )

    if not rows:
        raise RuntimeError("Nenhum modelo habilitado gerou resultados no benchmark.")

    results_dir.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    for col in ["metric_target_value", METRIC_NDCG_10, METRIC_MRR_10]:
        if col not in df.columns:
            df[col] = 0.0
    df = df.sort_values(
        by=["metric_target_value", METRIC_NDCG_10, METRIC_MRR_10],
        ascending=False,
        na_position="last",
    ).reset_index(drop=True)

    best_row = df.iloc[0]
    csv_path = results_dir / "benchmark_modelos.csv"
    json_path = results_dir / "benchmark_modelos.json"
    md_path = results_dir / "benchmark_modelos.md"

    df.to_csv(csv_path, index=False)
    json_rows = _jsonable_records(df)
    payload = {
        "generated_at_utc": now_iso(),
        "config_path": rel_path(config_path),
        "results_dir": rel_path(results_dir),
        "modelos_dir": rel_path(modelos_root),
        "split_signature_at_execution": split_signature_from_manifest_file(
            manifest_path(runtime_context.splits_dir)
        ),
        "unique_model_split_signatures": sorted(
            {
                str(row.get("model_split_signature", "")).strip()
                for row in json_rows
                if str(row.get("model_split_signature", "")).strip()
            }
        ),
        "best_model": json_rows[0],
        "rows": json_rows,
    }
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    md_path.write_text(_markdown_benchmark(df, best_row), encoding="utf-8")

    print("\n=== Benchmark concluído ===")
    print(f"CSV  : {csv_path}")
    print(f"JSON : {json_path}")
    print(f"MD   : {md_path}")
    print(
        f"Melhor modelo: {best_row['model_id']} "
        f"({best_row['metric_target']}={best_row['metric_target_value']:.4f})"
    )
    benchmark_progress.finish("Benchmark finalizado")


if __name__ == "__main__":
    main()
