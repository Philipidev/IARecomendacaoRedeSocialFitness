"""
Camada de servico nao-interativa sobre o pipeline.

Reutiliza toda a logica de main.py, expondo funcoes que recebem
parametros diretamente (sem input/ask_yes_no/choose_option).
O CLI (main.py) continua funcionando sem alteracoes.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Re-exportacoes diretas de main.py (funcoes que nao usam input)
# ---------------------------------------------------------------------------

from main import (  # noqa: F401 — re-exports
    ROOT,
    STATE_PATH,
    DATASET_DIR,
    DOWNLOAD_OPTIONS,
    DEFAULT_SPLIT_CONFIG,
    DOWNLOAD_SCRIPT,
    PIPELINE_SCRIPT,
    PREPARAR_SCRIPT,
    DIVIDIR_SCRIPT,
    TREINAR_SCRIPT,
    PREPARAR_LTR_SCRIPT,
    TREINAR_LTR_SCRIPT,
    AVALIAR_MODELO_SCRIPT,
    AVALIAR_POPULARIDADE_SCRIPT,
    AVALIACAO_MANUAL_SCRIPT,
    OTIMIZAR_PESOS_SCRIPT,
    BENCHMARK_TCC_SCRIPT,
    TCC_CONFIG_PATH,
    PESOS_OTIMOS_FILENAME,
    PESOS_EXPERIMENTOS_FILENAME,
    # Estado
    load_state,
    save_state,
    refresh_state,
    update_and_save,
    default_state,
    ensure_state_shape,
    # Artefatos
    build_file_status,
    build_stage_status,
    build_model_dir_status,
    badge,
    # TCC config
    load_tcc_config_safe,
    tcc_models_from_config,
    tcc_model_map,
    # Alvos
    default_model_target,
    make_experiment_model_target,
    normalize_model_target,
    default_benchmark_target,
    normalize_benchmark_target,
    resolve_model_target_dir,
    get_selected_model_target,
    # Labels
    selected_dataset_label,
    selected_model_target_label,
    model_target_label,
    benchmark_target_label,
    # Utilitarios
    now_iso,
    rel_path,
    abs_path,
    register_run,
    update_selected_dataset,
    current_dataset_context,
    extraction_matches_selected_dataset,
    get_last_run,
    discover_datasets,
    find_download_option,
    download_dataset_by_scale_factor,
    build_training_plan,
    merge_experiment_metadata,
    write_manual_baseline_weights,
    target_results_root,
    get_tcc_model_config,
)

from dataset_context import dataset_context


# ---------------------------------------------------------------------------
# Wrappers nao-interativos
# ---------------------------------------------------------------------------


def service_select_dataset(state: dict[str, Any], dataset_index: int) -> dict[str, Any]:
    """Seleciona dataset por indice (sem input)."""
    datasets = state["workspace"]["datasets"]
    if not datasets:
        raise ValueError("Nenhum dataset disponivel.")
    if dataset_index < 0 or dataset_index >= len(datasets):
        raise ValueError(f"Indice invalido: {dataset_index}. Disponiveis: 0-{len(datasets) - 1}")

    ds = datasets[dataset_index]
    update_selected_dataset(
        state,
        abs_path(ds["path"]) or (DATASET_DIR / ds["name"]),
        scale_factor=ds.get("scale_factor"),
        source="existente",
    )
    return update_and_save(state)


def service_select_model_target(state: dict[str, Any], target_index: int) -> dict[str, Any]:
    """Seleciona modelo/experimento alvo por indice (sem input).

    Indice 0 = modelo padrao.
    Indices >= 1 = experimentos do TCC config.
    """
    state = update_and_save(state)
    config, error = load_tcc_config_safe()
    dataset_key = (
        state.get("selected_dataset", {}).get("dataset_key")
        if isinstance(state.get("selected_dataset"), dict)
        else None
    )
    default_target = default_model_target(dataset_key)
    payloads: list[dict[str, Any]] = [default_target]

    if not error and config:
        for model_cfg in tcc_models_from_config(config, enabled_only=False):
            payloads.append(make_experiment_model_target(model_cfg, dataset_key=dataset_key))

    if target_index < 0 or target_index >= len(payloads):
        raise ValueError(f"Indice invalido: {target_index}. Disponiveis: 0-{len(payloads) - 1}")

    selected = dict(payloads[target_index])
    selected["selected_at"] = now_iso()
    state["selected_model_target"] = normalize_model_target(selected, config, dataset_key)
    return update_and_save(state)


def service_list_model_targets(state: dict[str, Any]) -> list[dict[str, Any]]:
    """Lista modelos/experimentos disponiveis com metadados."""
    config, error = load_tcc_config_safe()
    dataset_key = (
        state.get("selected_dataset", {}).get("dataset_key")
        if isinstance(state.get("selected_dataset"), dict)
        else None
    )
    default_target = default_model_target(dataset_key)
    targets = [{"index": 0, "label": f"Modelo padrao ({default_target['model_dir']})", **default_target}]

    if not error and config:
        for i, model_cfg in enumerate(tcc_models_from_config(config, enabled_only=False), start=1):
            target = make_experiment_model_target(model_cfg, dataset_key=dataset_key)
            enabled = bool(model_cfg.get("enabled", True))
            desc = str(model_cfg.get("descricao", "")).strip()
            label = (
                f"{model_cfg['id']} ({model_cfg.get('family', 'baseline_hibrido')})"
                f"{'' if enabled else ' [desabilitado]'}"
                f"{f' - {desc}' if desc else ''}"
            )
            targets.append({"index": i, "label": label, "enabled": enabled, **target})
    return targets


def service_select_benchmark(
    state: dict[str, Any],
    scope: str,
    model_ids: list[str] | None = None,
) -> dict[str, Any]:
    """Configura benchmark TCC (sem input)."""
    selection: dict[str, Any] = {"scope": scope, "model_ids": model_ids or []}
    if scope == "all":
        selection["model_ids"] = []
    selection["selected_at"] = now_iso()
    config, _ = load_tcc_config_safe()
    state["selected_benchmark"] = normalize_benchmark_target(selection, config)
    return update_and_save(state)


def service_get_eval_modes(family: str) -> list[dict[str, Any]]:
    """Retorna modos de avaliacao compativeis com a familia."""
    modes = [{"id": "offline", "label": "Avaliacao offline do recomendador"}]
    if family == "baseline_hibrido":
        modes.append({"id": "popularidade", "label": "Impacto do peso de popularidade"})
    modes.append({"id": "manual", "label": "Avaliacao manual reproduzivel"})
    if family == "baseline_hibrido":
        modes.append({"id": "otimizacao", "label": "Otimizacao de pesos"})
    all_modes = (
        ["offline", "popularidade", "manual"]
        if family == "baseline_hibrido"
        else ["offline", "manual"]
    )
    modes.append({"id": "all", "label": "Rodar todas as avaliacoes compativeis", "expands_to": all_modes})
    return modes


def service_get_state_details(state: dict[str, Any]) -> dict[str, Any]:
    """Retorna detalhes completos do estado para a UI."""
    workspace = state["workspace"]
    treinamento = workspace["treinamento"]
    avaliacao = workspace["avaliacao"]
    namespace = workspace.get("selected_dataset_context", {})

    return {
        "updated_at": state.get("updated_at"),
        "dataset_label": selected_dataset_label(state),
        "model_target_label": selected_model_target_label(state),
        "benchmark_label": benchmark_target_label(state),
        "selected_dataset": state.get("selected_dataset"),
        "selected_model_target": state.get("selected_model_target"),
        "selected_benchmark": state.get("selected_benchmark"),
        "namespace": namespace,
        "extraction_matches": extraction_matches_selected_dataset(state),
        "artifacts": {
            "extracao": {
                "ready": workspace["extracao"]["ready"],
                "existing": len(workspace["extracao"]["existing"]),
                "total": workspace["extracao"]["total"],
            },
            "dados": {"ready": treinamento["dados"]["ready"]},
            "splits": {"ready": treinamento["splits"]["ready"]},
            "modelo_padrao": {"ready": treinamento["modelo"]["ready"]},
            "modelo_alvo": {
                "ready": treinamento["alvo"]["required"]["ready"],
                "model_dir": treinamento["alvo"]["model_dir"],
                "family": treinamento["alvo"]["family"],
            },
        },
        "avaliacao": {
            "can_run": avaliacao["can_run"],
            "resultados": avaliacao.get("resultados", {}),
        },
        "tcc": workspace.get("tcc", {}),
        "last_runs": state.get("last_runs", {}),
        "datasets": workspace.get("datasets", []),
    }


# ---------------------------------------------------------------------------
# Builders de argumentos para subprocessos (usados pelo executor web)
# ---------------------------------------------------------------------------


def build_extraction_args(state: dict[str, Any]) -> tuple[Path, list[str]]:
    """Retorna (script, args) para rodar extracao."""
    selected = state.get("selected_dataset") or {}
    dataset_path = abs_path(selected.get("path"))
    if not dataset_path or not dataset_path.exists():
        raise ValueError("Dataset ativo nao encontrado ou nao selecionado.")

    args = ["--dataset-path", str(dataset_path)]
    if selected.get("dataset_key"):
        args.extend(["--dataset-key", str(selected["dataset_key"])])
    return PIPELINE_SCRIPT, args


def build_training_args(
    state: dict[str, Any],
    split_config: dict[str, Any] | None = None,
) -> list[tuple[str, Path, list[str]]]:
    """Retorna lista de (etapa_label, script, args) para treinamento completo.

    Sequencia: preparacao_dados → dividir_dataset → treinar → (otimizar|ltr_prep|ltr_train)
    """
    target = get_selected_model_target(state)
    if split_config is None:
        if str(target.get("type", "modelo_padrao")) == "experimento_tcc":
            split_config = None  # build_training_plan usara config do experimento
        else:
            split_config = dict(DEFAULT_SPLIT_CONFIG)

    training_plan, error = build_training_plan(target, split_config)
    if error:
        raise ValueError(error)
    assert training_plan is not None

    selected = state.get("selected_dataset") or {}
    dataset_key = selected.get("dataset_key") if isinstance(selected, dict) else None
    dataset_path = selected.get("path") if isinstance(selected, dict) else None
    scale_factor = selected.get("scale_factor") if isinstance(selected, dict) else None

    steps: list[tuple[str, Path, list[str]]] = []

    # 1. Preparacao de dados
    prep_args: list[str] = []
    if dataset_key:
        prep_args.extend(["--dataset-key", str(dataset_key)])
    if dataset_path:
        prep_args.extend(["--dataset-path", str(dataset_path)])
    if scale_factor:
        prep_args.extend(["--scale-factor", str(scale_factor)])
    steps.append(("Preparacao de dados", PREPARAR_SCRIPT, prep_args))

    # 2. Split
    sc = training_plan["split_config"]
    split_args = [
        "--train", str(sc["train"]),
        "--val", str(sc["val"]),
        "--test", str(sc["test"]),
        "--seed", str(sc["seed"]),
    ]
    if dataset_key:
        split_args.extend(["--dataset-key", str(dataset_key)])
    if dataset_path:
        split_args.extend(["--dataset-path", str(dataset_path)])
    if scale_factor:
        split_args.extend(["--scale-factor", str(scale_factor)])
    steps.append(("Dividir dataset", DIVIDIR_SCRIPT, split_args))

    # 3. Treinar
    train_args = ["--model-dir", str(training_plan["model_dir"])]
    if training_plan["experiment_id"]:
        train_args.extend(["--experiment-id", str(training_plan["experiment_id"])])
    training_cfg = training_plan["training_cfg"]
    if training_cfg.get("dataset_completo"):
        train_args.append("--dataset-completo")
    elif training_cfg.get("catalogo_completo", bool(training_plan["experiment_id"])):
        train_args.append("--catalogo-completo")
    if dataset_key:
        train_args.extend(["--dataset-key", str(dataset_key)])
    if dataset_path:
        train_args.extend(["--dataset-path", str(dataset_path)])
    if scale_factor:
        train_args.extend(["--scale-factor", str(scale_factor)])
    steps.append(("Treinar modelo", TREINAR_SCRIPT, train_args))

    # 4. Pos-treinamento (familia-dependente)
    if training_plan["family"] == "baseline_hibrido":
        params = training_plan["params"]
        if params.get("usar_pesos_otimos"):
            otim_args = [
                "--model-dir", str(training_plan["model_dir"]),
                "--grid-step", str(params.get("grid_step", 0.1)),
                "--random-search", str(params.get("random_search", 0)),
                "--top-k", str(params.get("otimizacao_top_k", 10)),
                "--max-queries", str(params.get("max_queries_otimizacao", 300)),
                "--seed", str(training_plan["split_config"]["seed"]),
            ]
            if training_plan["experiment_id"]:
                otim_args.extend([
                    "--out-csv", str(target_results_root(target, dataset_key) / PESOS_EXPERIMENTOS_FILENAME),
                    "--out-json", str(training_plan["model_dir"] / PESOS_OTIMOS_FILENAME),
                ])
            if dataset_key:
                otim_args.extend(["--dataset-key", str(dataset_key)])
            steps.append(("Otimizar pesos", OTIMIZAR_PESOS_SCRIPT, otim_args))

    if training_plan["family"] == "ltr_lightgbm":
        dataset_cfg = training_plan["dataset_ltr"]
        features_enabled = dataset_cfg.get("features_enabled", [])
        ltr_prep_args = [
            "--model-dir", str(training_plan["model_dir"]),
            "--train-out", str(training_plan["model_dir"] / "ltr_train.parquet"),
            "--val-out", str(training_plan["model_dir"] / "ltr_val.parquet"),
            "--meta-out", str(training_plan["model_dir"] / "ltr_dataset_meta.json"),
            "--negatives-per-query", str(dataset_cfg.get("negatives_per_query", 50)),
            "--hard-negative-topn", str(dataset_cfg.get("hard_negative_topn", 500)),
            "--max-queries-train", str(dataset_cfg.get("max_queries_train", 500)),
            "--max-queries-val", str(dataset_cfg.get("max_queries_val", 200)),
            "--seed", str(dataset_cfg.get("seed", training_plan["split_config"]["seed"])),
        ]
        if dataset_key:
            ltr_prep_args.extend(["--dataset-key", str(dataset_key)])
        if isinstance(features_enabled, list) and features_enabled:
            ltr_prep_args.extend(["--features", *[str(f) for f in features_enabled]])
        steps.append(("Preparar dataset LTR", PREPARAR_LTR_SCRIPT, ltr_prep_args))

        params = training_plan["params"]
        ltr_train_args = [
            "--model-dir", str(training_plan["model_dir"]),
            "--train-dataset", str(training_plan["model_dir"] / "ltr_train.parquet"),
            "--val-dataset", str(training_plan["model_dir"] / "ltr_val.parquet"),
            "--meta-dataset", str(training_plan["model_dir"] / "ltr_dataset_meta.json"),
            "--objective", str(params.get("objective", "lambdarank")),
            "--metric-at", *[str(k) for k in params.get("metric_at", training_plan["top_k"])],
            "--num-leaves", str(params.get("num_leaves", 31)),
            "--learning-rate", str(params.get("learning_rate", 0.05)),
            "--n-estimators", str(params.get("n_estimators", 300)),
            "--min-data-in-leaf", str(params.get("min_data_in_leaf", 20)),
            "--feature-fraction", str(params.get("feature_fraction", 0.9)),
            "--bagging-fraction", str(params.get("bagging_fraction", 0.8)),
            "--bagging-freq", str(params.get("bagging_freq", 1)),
            "--seed", str(params.get("seed", training_plan["split_config"]["seed"])),
        ]
        steps.append(("Treinar LTR", TREINAR_LTR_SCRIPT, ltr_train_args))

    return steps


def build_evaluation_args(
    state: dict[str, Any],
    modes: list[str],
) -> list[tuple[str, Path, list[str]]]:
    """Retorna lista de (etapa_label, script, args) para avaliacao."""
    target = get_selected_model_target(state)
    family = str(target.get("family", "baseline_hibrido"))
    selected = state.get("selected_dataset") or {}
    dataset_key = selected.get("dataset_key") if isinstance(selected, dict) else None
    model_dir = resolve_model_target_dir(target)
    results_root = target_results_root(target, dataset_key)

    # Expandir "all"
    if "all" in modes:
        modes = (
            ["offline", "popularidade", "manual"]
            if family == "baseline_hibrido"
            else ["offline", "manual"]
        )

    # Validar compatibilidade
    unsupported = [m for m in modes if m in {"popularidade", "otimizacao"} and family != "baseline_hibrido"]
    if unsupported:
        raise ValueError(
            f"Modos nao suportados para {family}: {', '.join(unsupported)}"
        )

    steps: list[tuple[str, Path, list[str]]] = []

    for mode in modes:
        if mode == "offline":
            args = ["--model-dir", str(model_dir), "--k", "5", "10", "20"]
            if dataset_key:
                args.extend(["--dataset-key", str(dataset_key)])
            args.extend(["--out-dir", str(results_root / "offline")])
            steps.append(("Avaliacao offline", AVALIAR_MODELO_SCRIPT, args))

        elif mode == "popularidade":
            args = [
                "--model-dir", str(model_dir),
                "--k", "10",
                "--peso-depois", "0.10",
                "--out-json", str(results_root / "popularidade" / "metricas_antes_depois.json"),
            ]
            if dataset_key:
                args.extend(["--dataset-key", str(dataset_key)])
            steps.append(("Avaliacao popularidade", AVALIAR_POPULARIDADE_SCRIPT, args))

        elif mode == "manual":
            args = [
                "--model-dir", str(model_dir),
                "--saida", str(results_root / "manual" / "avaliacao_manual.md"),
                "--saida-json", str(results_root / "manual" / "avaliacao_manual.json"),
            ]
            steps.append(("Avaliacao manual", AVALIACAO_MANUAL_SCRIPT, args))

        elif mode == "otimizacao":
            args = [
                "--model-dir", str(model_dir),
                "--grid-step", "0.1",
                "--top-k", "10",
                "--max-queries", "300",
                "--seed", "42",
                "--out-csv", str(results_root / PESOS_EXPERIMENTOS_FILENAME),
                "--out-json", str(model_dir / PESOS_OTIMOS_FILENAME),
            ]
            if dataset_key:
                args.extend(["--dataset-key", str(dataset_key)])
            steps.append(("Otimizacao de pesos", OTIMIZAR_PESOS_SCRIPT, args))

    return steps


def build_benchmark_args(state: dict[str, Any]) -> tuple[Path, list[str]]:
    """Retorna (script, args) para benchmark TCC."""
    selected = state.get("selected_dataset") or {}
    dataset_key = selected.get("dataset_key") if isinstance(selected, dict) else None
    dataset_path = selected.get("path") if isinstance(selected, dict) else None
    scale_factor = selected.get("scale_factor") if isinstance(selected, dict) else None

    args = ["--config", str(TCC_CONFIG_PATH)]
    if dataset_key:
        args.extend(["--dataset-key", str(dataset_key)])
    if dataset_path:
        args.extend(["--dataset-path", str(dataset_path)])
    if scale_factor:
        args.extend(["--scale-factor", str(scale_factor)])

    benchmark_target = state.get("selected_benchmark", default_benchmark_target())
    if (
        isinstance(benchmark_target, dict)
        and benchmark_target.get("scope") == "subset"
        and benchmark_target.get("model_ids")
    ):
        args.extend(["--model-ids", *[str(mid) for mid in benchmark_target["model_ids"]]])

    return BENCHMARK_TCC_SCRIPT, args
