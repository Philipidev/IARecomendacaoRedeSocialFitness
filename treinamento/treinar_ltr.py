from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from time import perf_counter

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from progress_utils import IterationProgress, StageProgress
from treinamento.model_utils import merge_model_metadata, rel_path, resolve_model_dir


def _resolve_path(path_str: str | None, default_path: Path) -> Path:
    if not path_str:
        return default_path
    path = Path(path_str)
    return path if path.is_absolute() else ROOT / path


def _group_sizes(df: pd.DataFrame) -> list[int]:
    return (
        df.groupby("query_id", sort=True)
        .size()
        .astype(int)
        .tolist()
    )


def _load_feature_meta(meta_path: Path) -> dict:
    if not meta_path.exists():
        raise FileNotFoundError(f"Metadata LTR ausente: {meta_path}")
    payload = json.loads(meta_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Metadata LTR inválida em {meta_path}")
    return payload


def _progress_callback(total_rounds: int, label: str):
    progress = IterationProgress(
        total=total_rounds,
        label=label,
        every_percent=5,
    )
    progress.start("Rounds concluídos")

    def callback(env) -> None:
        progress.log(env.iteration + 1, detail="Rounds concluídos")

    callback.order = 15
    callback.before_iteration = False
    callback.progress = progress
    return callback


def main() -> None:
    parser = argparse.ArgumentParser(description="Treina um LightGBMRanker para ranking.")
    parser.add_argument("--model-dir", type=str, default="treinamento/modelo")
    parser.add_argument("--train-dataset", type=str, default=None)
    parser.add_argument("--val-dataset", type=str, default=None)
    parser.add_argument("--meta-dataset", type=str, default=None)
    parser.add_argument("--objective", type=str, default="lambdarank")
    parser.add_argument("--metric-at", nargs="+", type=int, default=[5, 10, 20])
    parser.add_argument("--num-leaves", type=int, default=31)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--n-estimators", type=int, default=300)
    parser.add_argument("--min-data-in-leaf", type=int, default=20)
    parser.add_argument("--feature-fraction", type=float, default=0.9)
    parser.add_argument("--bagging-fraction", type=float, default=0.8)
    parser.add_argument("--bagging-freq", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    try:
        import lightgbm as lgb  # type: ignore
    except ModuleNotFoundError as exc:  # pragma: no cover - depende do ambiente
        raise ModuleNotFoundError(
            "lightgbm não instalado. Execute: pip install -r requirements.txt"
        ) from exc

    model_dir = resolve_model_dir(args.model_dir)
    train_dataset = _resolve_path(args.train_dataset, model_dir / "ltr_train.parquet")
    val_dataset = _resolve_path(args.val_dataset, model_dir / "ltr_val.parquet")
    meta_dataset = _resolve_path(args.meta_dataset, model_dir / "ltr_dataset_meta.json")

    stage_progress = StageProgress(
        total_stages=4,
        label=f"Treino LTR {model_dir.name}",
    )

    stage_progress.step("Carregando datasets LTR")
    train_df = pd.read_parquet(train_dataset)
    val_df = pd.read_parquet(val_dataset) if val_dataset.exists() else pd.DataFrame()
    meta = _load_feature_meta(meta_dataset)
    feature_columns = list(meta.get("feature_columns", []))
    if not feature_columns:
        raise ValueError("feature_columns não encontrado em ltr_dataset_meta.json")

    if train_df.empty:
        raise ValueError("Dataset de treino LTR vazio.")

    X_train = train_df[feature_columns]
    y_train = train_df["label"].astype(int)
    group_train = _group_sizes(train_df)

    print()
    stage_progress.step("Preparando datasets LightGBM")
    train_set = lgb.Dataset(X_train, label=y_train, group=group_train, free_raw_data=False)

    valid_sets = [train_set]
    valid_names = ["train"]
    group_val: list[int] | None = None
    if not val_df.empty:
        X_val = val_df[feature_columns]
        y_val = val_df["label"].astype(int)
        group_val = _group_sizes(val_df)
        val_set = lgb.Dataset(X_val, label=y_val, group=group_val, free_raw_data=False)
        valid_sets.append(val_set)
        valid_names.append("val")

    params = {
        "objective": args.objective,
        "metric": "ndcg",
        "ndcg_eval_at": args.metric_at,
        "num_leaves": args.num_leaves,
        "learning_rate": args.learning_rate,
        "min_data_in_leaf": args.min_data_in_leaf,
        "feature_fraction": args.feature_fraction,
        "bagging_fraction": args.bagging_fraction,
        "bagging_freq": args.bagging_freq,
        "seed": args.seed,
        "verbosity": -1,
        "deterministic": True,
        "force_col_wise": True,
    }

    callbacks = [lgb.log_evaluation(period=25)]
    if not val_df.empty:
        callbacks.append(lgb.early_stopping(stopping_rounds=30, verbose=True))
    train_progress = _progress_callback(args.n_estimators, f"Rounds LTR {model_dir.name}")
    callbacks.append(train_progress)

    print()
    stage_progress.step("Treinando LightGBMRanker")
    start = perf_counter()
    booster = lgb.train(
        params=params,
        train_set=train_set,
        num_boost_round=args.n_estimators,
        valid_sets=valid_sets,
        valid_names=valid_names,
        callbacks=callbacks,
    )
    elapsed_s = perf_counter() - start
    train_progress.progress.finish("Treino finalizado")

    print()
    stage_progress.step("Salvando artefatos e metadata")
    model_dir.mkdir(parents=True, exist_ok=True)
    model_path = model_dir / "ltr_model.txt"
    booster.save_model(str(model_path))

    importance_df = pd.DataFrame(
        {
            "feature": feature_columns,
            "importance_gain": booster.feature_importance(importance_type="gain"),
            "importance_split": booster.feature_importance(importance_type="split"),
        }
    ).sort_values("importance_gain", ascending=False)
    importance_path = model_dir / "ltr_feature_importance.csv"
    importance_df.to_csv(importance_path, index=False)

    schema_path = model_dir / "ltr_feature_schema.json"
    schema_payload = {
        "feature_columns": feature_columns,
        "categorical_maps": meta.get("categorical_maps", {}),
        "dataset_meta_path": rel_path(meta_dataset),
        "model_path": rel_path(model_path),
    }
    schema_path.write_text(json.dumps(schema_payload, ensure_ascii=False, indent=2), encoding="utf-8")

    best_iteration = int(booster.best_iteration or args.n_estimators)
    metadata_payload = {
        "family": "ltr_lightgbm",
        "model_dir": rel_path(model_dir),
        "ltr": {
            "train_dataset": rel_path(train_dataset),
            "val_dataset": rel_path(val_dataset) if val_dataset.exists() else None,
            "feature_columns": feature_columns,
            "params": {
                "objective": args.objective,
                "metric_at": args.metric_at,
                "num_leaves": args.num_leaves,
                "learning_rate": args.learning_rate,
                "n_estimators": args.n_estimators,
                "min_data_in_leaf": args.min_data_in_leaf,
                "feature_fraction": args.feature_fraction,
                "bagging_fraction": args.bagging_fraction,
                "bagging_freq": args.bagging_freq,
                "seed": args.seed,
            },
            "best_iteration": best_iteration,
            "tempo_treinamento_s": elapsed_s,
            "feature_importance_path": rel_path(importance_path),
        },
    }
    merge_model_metadata(model_dir, metadata_payload)

    print("\n=== Treino LTR concluído ===")
    print(f"Modelo      : {model_path}")
    print(f"Schema      : {schema_path}")
    print(f"Importância : {importance_path}")
    print(f"Melhor iter.: {best_iteration}")
    print(f"Tempo (s)   : {elapsed_s:.2f}")


if __name__ == "__main__":
    main()
