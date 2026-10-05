#!/usr/bin/env python3
"""Train and evaluate an ensemble of two HistGradientBoosting regressors.

Ensemble members
----------------
* Structured model: HistGradientBoostingRegressor trained on the wide tabular matrix
  from build_reach_features.py (data/features/features_rich.csv), excluding `llm_*`.
* LLM model: HistGradientBoostingRegressor trained only on the LLM-extracted columns
  from build_llm_extractions.py (data/features/features_llm.csv, `llm_*`). The LLM's
  own `price_label` is dropped - it would leak the target.

Both members are trained on identical GroupKFold splits and their predictions are
blended on the log-price scale: `w * structured + (1 - w) * llm`. The weight defaults
to 0.5 and is configurable with --blend-weight; the report also shows the out-of-fold
optimal weight as a diagnostic (it is fitted on the same OOF rows, so it is reported
separately from the headline metrics).

Target:  asking_price_rub, trained on the log scale and scored on the ruble scale.
CV:      GroupKFold by `group_locality` so listings from one locality never split
         between train and test; use --cv kfold only for a quicker comparison.

Key design points
-----------------
1. Missing values are NOT dropped: HistGradientBoosting consumes NaN directly, which
   matters because both the structured matrix and the LLM block are sparse.
2. The target is modelled on the log scale (price skew ≈ 9.8) and back-transformed
   with expm1 for the ruble-scale metrics, with a clip guard against extrapolation.
3. Grouped CV keeps near-duplicate listings of one locality out of the test set, and
   both members see exactly the same folds - required for a fair blend.
4. LLM features are joined strictly on `listing_id`.
5. Results go to reports/histgb_reach+llm/, never into another model's report directory.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    mean_absolute_error,
    mean_absolute_percentage_error,
    median_absolute_error,
    r2_score,
    root_mean_squared_error,
)
from sklearn.model_selection import GroupKFold, KFold, ParameterSampler


LOG = logging.getLogger("domclick.ensemble.llm")
IDENTIFIER_COLUMN = "listing_id"
TARGET_COLUMN = "asking_price_rub"
LOG_TARGET_COLUMN = "log_asking_price_rub"
GROUP_COLUMN = "group_locality"

# id, both target variants and the CV group key must never enter the design matrix.
NON_FEATURE_COLUMNS = (IDENTIFIER_COLUMN, TARGET_COLUMN, LOG_TARGET_COLUMN, GROUP_COLUMN)

# LLM-extracted columns are prefixed to keep them distinguishable from the structured ones.
LLM_FEATURE_PREFIX = "llm_"
# The LLM's own price guess is the target in disguise; joining it would leak the label.
LEAKY_LLM_COLUMNS = ("price_label",)

# Ensemble members plus the blended prediction, used as labels in the metrics tables.
STRUCTURED_MODEL = "hgb_structured"
LLM_MODEL = "hgb_llm"
ENSEMBLE_MODEL = "ensemble"
MODEL_NAMES = (STRUCTURED_MODEL, LLM_MODEL, ENSEMBLE_MODEL)
MODEL_LABELS = {
    STRUCTURED_MODEL: "HistGB (структурные)",
    LLM_MODEL: "HistGB (LLM-признаки)",
    ENSEMBLE_MODEL: "Ансамбль",
}

# Metrics on the ruble scale plus one on the modelled log scale.
METRIC_COLUMNS = ("r2_log", "r2_rub", "rmse_rub", "mae_rub", "median_ae_rub", "mape_pct")

PARAM_DISTRIBUTIONS = {
    "learning_rate": [0.03, 0.05, 0.08, 0.12],
    "max_leaf_nodes": [15, 31, 63],
    "min_samples_leaf": [10, 20, 40],
    "l2_regularization": [0.0, 1.0, 10.0],
}


def default_input_path() -> Path:
    """Where build_reach_features.py writes the modeling matrix."""
    return Path(__file__).resolve().parent / "data" / "features" / "features_rich.csv"


def default_llm_input_path() -> Path:
    """Where build_llm_extractions.py writes the LLM-extracted features."""
    return Path(__file__).resolve().parent / "data" / "features" / "features_llm.csv"


def default_report_dir() -> Path:
    """Reports go to reports/, in a directory named after this model."""
    return Path(__file__).resolve().parent / "reports" / "histgb_reach+llm"


def merge_llm_features(frame: pd.DataFrame, llm_path: Path) -> pd.DataFrame:
    """Left-join the LLM-extracted features onto the structured matrix by `listing_id`.

    The join is strict: every LLM column must be new (the `llm_` prefix guarantees
    that) and listings absent from the LLM file simply keep NaN, which the model
    handles natively.
    """
    try:
        llm = pd.read_csv(llm_path)
    except FileNotFoundError as exc:
        raise SystemExit(
            f"Не найден файл LLM-признаков: {llm_path}\n"
            "Сначала запустите build_llm_extractions.py или передайте --no-llm-features."
        ) from exc
    except OSError as exc:
        raise SystemExit(f"Не удалось прочитать {llm_path}: {exc}") from exc

    if IDENTIFIER_COLUMN not in llm.columns:
        raise SystemExit(f"В {llm_path} нет колонки {IDENTIFIER_COLUMN}.")
    if IDENTIFIER_COLUMN not in frame.columns:
        raise SystemExit(f"В основной матрице нет колонки {IDENTIFIER_COLUMN} для объединения.")

    leaky = [column for column in llm.columns if column in LEAKY_LLM_COLUMNS]
    if leaky:
        LOG.warning("LLM-колонки, исключённые как утечка цели: %s", ", ".join(leaky))
        llm = llm.drop(columns=leaky)

    if llm[IDENTIFIER_COLUMN].duplicated().any():
        duplicates = int(llm[IDENTIFIER_COLUMN].duplicated().sum())
        raise SystemExit(
            f"В {llm_path} {duplicates} повторяющихся {IDENTIFIER_COLUMN}; ожидается уникальный ключ."
        )

    llm_columns = [column for column in llm.columns if column != IDENTIFIER_COLUMN]
    collisions = [column for column in llm_columns if column in frame.columns]
    if collisions:
        raise SystemExit(
            "LLM-признаки конфликтуют с существующими колонками: " + ", ".join(collisions)
        )

    merged = frame.merge(llm, on=IDENTIFIER_COLUMN, how="left", validate="many_to_one")
    matched = int(merged[llm_columns].notna().any(axis=1).sum()) if llm_columns else 0
    LOG.info(
        "LLM-признаков добавлено: %s; строк с хотя бы одним значением: %s из %s",
        len(llm_columns), matched, len(merged),
    )
    if matched < len(merged):
        LOG.warning(
            "У %s строк LLM-признаки отсутствуют (останутся NaN)", len(merged) - matched
        )
    return merged


def load_data(path: Path, llm_path: Path | None = None
              ) -> tuple[pd.DataFrame, pd.Series, pd.Series, pd.Series, list[str]]:
    """Read the matrix and split it into features, target, CV groups and identifiers.

    When `llm_path` is given the LLM columns are left-joined on `listing_id` first.
    Missing values are intentionally preserved: the model handles them natively.
    """
    try:
        frame = pd.read_csv(path)
    except FileNotFoundError as exc:
        raise SystemExit(
            f"Не найден файл признаков: {path}\nСначала запустите build_reach_features.py."
        ) from exc
    except OSError as exc:
        raise SystemExit(f"Не удалось прочитать {path}: {exc}") from exc

    if TARGET_COLUMN not in frame.columns:
        raise SystemExit(f"В {path} нет целевой колонки {TARGET_COLUMN}.")

    if llm_path is not None:
        frame = merge_llm_features(frame, llm_path)

    feature_columns = [
        column for column in frame.columns if column not in NON_FEATURE_COLUMNS
    ]
    if not feature_columns:
        raise SystemExit(f"В {path} нет признаков кроме идентификатора, цели и ключа групп.")

    for column in (*feature_columns, TARGET_COLUMN):
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
        frame[column] = frame[column].replace([np.inf, -np.inf], np.nan)

    unusable = frame[TARGET_COLUMN].isna() | frame[TARGET_COLUMN].le(0)
    if unusable.any():
        LOG.warning("Строк без положительной цены отброшено: %s", int(unusable.sum()))
        frame = frame.loc[~unusable]
    if frame.empty:
        raise SystemExit("Не осталось строк с положительной ценой.")

    frame = frame.reset_index(drop=True)
    identifiers = (
        frame[IDENTIFIER_COLUMN] if IDENTIFIER_COLUMN in frame.columns
        else pd.Series(np.arange(len(frame)), dtype="int64")
    )
    groups = (
        frame[GROUP_COLUMN].fillna(-1).astype("int64")
        if GROUP_COLUMN in frame.columns
        else pd.Series(-1, index=frame.index, dtype="int64")
    )
    missing_cells = int(frame[feature_columns].isna().sum().sum())
    LOG.info("Пропусков в матрице признаков: %s (не заполняются)", missing_cells)
    return frame[feature_columns], frame[TARGET_COLUMN], groups, identifiers, feature_columns


def split_feature_columns(feature_columns: list[str]) -> tuple[list[str], list[str]]:
    """Split columns into the structured block and the `llm_*` block."""
    llm = [column for column in feature_columns if column.startswith(LLM_FEATURE_PREFIX)]
    structured = [
        column for column in feature_columns if not column.startswith(LLM_FEATURE_PREFIX)
    ]
    return structured, llm


def make_model(params: dict[str, Any], seed: int, max_iter: int) -> HistGradientBoostingRegressor:
    """Gradient boosting with early stopping on an internal validation split."""
    return HistGradientBoostingRegressor(
        max_iter=max_iter,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=20,
        random_state=seed,
        **params,
    )


def make_splitter(kind: str, folds: int, seed: int):
    """Grouped split (default) or plain KFold for comparison."""
    if kind == "group":
        return GroupKFold(n_splits=folds)
    return KFold(n_splits=folds, shuffle=True, random_state=seed)


def split_iterator(splitter, frame: pd.DataFrame, groups: pd.Series, kind: str):
    """Yield (train_index, test_index) for either splitter."""
    if kind == "group":
        return splitter.split(frame, groups=groups)
    return splitter.split(frame)


def drop_all_missing_columns(x_train: pd.DataFrame, x_test: pd.DataFrame
                             ) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Drop columns that are entirely NaN in the training part of the fold.

    Some very sparse LLM fields (e.g. a single digitised area) can happen to hold
    values only in the held-out locality of a grouped fold. sklearn's histogram
    binner raises on an all-NaN column, so such columns are removed from both
    parts; inspecting only the training part keeps the fold honest.
    """
    keep = x_train.notna().any(axis=0)
    if keep.all():
        return x_train, x_test
    dropped = list(x_train.columns[~keep])
    LOG.info("Колонки, полностью пустые в обучении фолда, исключены: %s", ", ".join(dropped))
    return x_train.loc[:, keep], x_test.loc[:, keep]


def score_fold(actual_rub: np.ndarray, predicted_rub: np.ndarray,
               actual_log: np.ndarray, predicted_log: np.ndarray) -> dict[str, float]:
    """Metrics on both scales: rubles for reporting, log for model comparison."""
    return {
        "r2_log": float(r2_score(actual_log, predicted_log)),
        "r2_rub": float(r2_score(actual_rub, predicted_rub)),
        "rmse_rub": float(root_mean_squared_error(actual_rub, predicted_rub)),
        "mae_rub": float(mean_absolute_error(actual_rub, predicted_rub)),
        "median_ae_rub": float(median_absolute_error(actual_rub, predicted_rub)),
        "mape_pct": float(mean_absolute_percentage_error(actual_rub, predicted_rub) * 100),
    }


def tune_params(x_train: pd.DataFrame, y_train_log: pd.Series, groups_train: pd.Series,
                n_candidates: int, inner_folds: int, seed: int,
                max_iter: int) -> tuple[dict[str, Any], pd.DataFrame]:
    """Random search on an inner grouped split of the training part only.

    The outer test fold is never touched, so the reported CV score stays honest.
    """
    inner = GroupKFold(n_splits=inner_folds)
    candidates = list(ParameterSampler(PARAM_DISTRIBUTIONS, n_iter=n_candidates,
                                      random_state=seed))
    rows: list[dict[str, Any]] = []
    for index, params in enumerate(candidates, start=1):
        scores: list[float] = []
        for train_index, valid_index in inner.split(x_train, groups=groups_train):
            model = make_model(params, seed, max_iter)
            model.fit(x_train.iloc[train_index], y_train_log.iloc[train_index])
            predicted = model.predict(x_train.iloc[valid_index])
            scores.append(root_mean_squared_error(
                y_train_log.iloc[valid_index], predicted
            ))
        rows.append({"candidate": index, "rmse_log_mean": float(np.mean(scores)), **params})
        print(f"  кандидат {index}/{len(candidates)}: RMSE(log) = {rows[-1]['rmse_log_mean']:.4f}")

    table = pd.DataFrame(rows).sort_values("rmse_log_mean").reset_index(drop=True)
    best = {name: table.loc[0, name] for name in PARAM_DISTRIBUTIONS}
    for name, values in PARAM_DISTRIBUTIONS.items():
        if all(isinstance(value, int) for value in values):
            best[name] = int(best[name])
        else:
            best[name] = float(best[name])
    return best, table


def mean_importance(frames: list[pd.Series]) -> pd.DataFrame:
    """Average per-fold permutation importance into one sorted feature table."""
    if not frames:
        return pd.DataFrame()
    return (
        pd.concat(frames, axis=1).mean(axis=1)
        .rename("importance_mean_r2_log").reset_index()
        .rename(columns={"index": "feature"})
        .sort_values("importance_mean_r2_log", ascending=False)
        .reset_index(drop=True)
    )


def cross_validate(x: pd.DataFrame, y: pd.Series, groups: pd.Series,
                   structured_columns: list[str], llm_columns: list[str],
                   args: argparse.Namespace
                   ) -> tuple[pd.DataFrame, dict[str, np.ndarray], dict[str, np.ndarray],
                              np.ndarray, dict[str, pd.DataFrame], pd.DataFrame]:
    """Train both ensemble members on identical folds and blend their predictions.

    Both members are HistGradientBoostingRegressor models: one sees the structured
    matrix, the other only the LLM-extracted columns. Returns per-fold metrics in long
    format, out-of-fold predictions in rubles and on the log scale for every model, the
    baseline predictions, permutation importances per member and the tuning table.
    """
    y_log = np.log(y.to_numpy(dtype=float))
    use_llm = bool(llm_columns)
    splitter = make_splitter(args.cv, args.folds, args.seed)
    rows: list[dict[str, Any]] = []
    predicted_rub = {name: np.full(len(x), np.nan) for name in MODEL_NAMES}
    predicted_log = {name: np.full(len(x), np.nan) for name in MODEL_NAMES}
    baseline_predictions = np.full(len(x), np.nan)
    importance_frames: dict[str, list[pd.Series]] = {STRUCTURED_MODEL: [], LLM_MODEL: []}
    tuning_frames: list[pd.DataFrame] = []

    for fold, (train_index, test_index) in enumerate(
            split_iterator(splitter, x, groups, args.cv), start=1):
        y_train_log = pd.Series(y_log[train_index], index=x.index[train_index])
        y_test_log = y_log[test_index]
        actual_rub = y.iloc[test_index].to_numpy(dtype=float)
        low, high = y.iloc[train_index].min(), y.iloc[train_index].max()

        # --- member 1: HistGB on the structured matrix ---
        x_struct_train = x.iloc[train_index][structured_columns]
        x_struct_test = x.iloc[test_index][structured_columns]
        x_struct_train, x_struct_test = drop_all_missing_columns(x_struct_train, x_struct_test)
        if not x_struct_train.shape[1]:
            raise SystemExit("После отбрасывания пустых колонок у структурной модели не осталось признаков.")

        params: dict[str, Any] = {}
        if args.tune:
            params, tuning = tune_params(
                x_struct_train, y_train_log, groups.iloc[train_index],
                args.tune, args.inner_folds, args.seed + fold, args.max_iter,
            )
            tuning.insert(0, "fold", fold)
            tuning_frames.append(tuning)

        structured_model = make_model(params, args.seed, args.max_iter)
        structured_model.fit(x_struct_train, y_train_log)
        fold_log: dict[str, np.ndarray] = {
            STRUCTURED_MODEL: structured_model.predict(x_struct_test)
        }
        iterations: dict[str, int | None] = {STRUCTURED_MODEL: int(structured_model.n_iter_)}

        # --- member 2: HistGB on the LLM-extracted columns only ---
        llm_model: HistGradientBoostingRegressor | None = None
        x_llm_test: pd.DataFrame | None = None
        if use_llm:
            x_llm_train = x.iloc[train_index][llm_columns]
            x_llm_test = x.iloc[test_index][llm_columns]
            x_llm_train, x_llm_test = drop_all_missing_columns(x_llm_train, x_llm_test)
            if x_llm_train.shape[1]:
                llm_model = make_model({}, args.seed + fold, args.max_iter)
                llm_model.fit(x_llm_train, y_train_log)
                fold_log[LLM_MODEL] = llm_model.predict(x_llm_test)
                iterations[LLM_MODEL] = int(llm_model.n_iter_)
            else:
                LOG.warning("Фолд %s: все LLM-колонки пусты в обучении, LLM-модель пропущена",
                            fold)

        # --- blend on the log scale: w * structured + (1 - w) * llm ---
        if LLM_MODEL in fold_log:
            weight = args.blend_weight
            fold_log[ENSEMBLE_MODEL] = (
                weight * fold_log[STRUCTURED_MODEL]
                + (1.0 - weight) * fold_log[LLM_MODEL]
            )
        else:
            fold_log[ENSEMBLE_MODEL] = fold_log[STRUCTURED_MODEL]
        iterations[ENSEMBLE_MODEL] = None

        fold_rows: dict[str, dict[str, Any]] = {}
        for name, log_prediction in fold_log.items():
            rub = np.expm1(log_prediction)
            clipped = 0
            # Trees cannot extrapolate far, but the log back-transform still can; clip to
            # the range seen in training so one wild prediction cannot dominate the metrics.
            if args.clip_predictions:
                clipped_values = np.clip(rub, low, high)
                clipped = int(np.sum(clipped_values != rub))
                rub = clipped_values
            predicted_log[name][test_index] = log_prediction
            predicted_rub[name][test_index] = rub
            row: dict[str, Any] = {
                "fold": fold,
                "model": name,
                "train_size": len(train_index),
                "test_size": len(test_index),
                "n_iter": iterations[name],
                "clipped": clipped,
            }
            row.update(score_fold(actual_rub, rub, y_test_log, log_prediction))
            rows.append(row)
            fold_rows[name] = row

        baseline = DummyRegressor(strategy="mean")
        baseline.fit(x_struct_train, y_train_log)
        baseline_predictions[test_index] = np.expm1(baseline.predict(x_struct_test))

        ensemble_r2 = fold_rows[ENSEMBLE_MODEL]["r2_rub"]
        if LLM_MODEL in fold_rows:
            print(f"Фолд {fold}/{args.folds}: R² в рублях — ансамбль {ensemble_r2:.3f}, "
                  f"структурный {fold_rows[STRUCTURED_MODEL]['r2_rub']:.3f}, "
                  f"LLM {fold_rows[LLM_MODEL]['r2_rub']:.3f}")
        else:
            print(f"Фолд {fold}/{args.folds}: R² в рублях = {ensemble_r2:.3f}")

        if args.importance_repeats:
            structured_result = permutation_importance(
                structured_model, x_struct_test, y_test_log,
                n_repeats=args.importance_repeats, random_state=args.seed,
            )
            importance_frames[STRUCTURED_MODEL].append(
                pd.Series(structured_result.importances_mean, index=x_struct_test.columns)
            )
            if llm_model is not None and x_llm_test is not None:
                llm_result = permutation_importance(
                    llm_model, x_llm_test, y_test_log,
                    n_repeats=args.importance_repeats, random_state=args.seed,
                )
                importance_frames[LLM_MODEL].append(
                    pd.Series(llm_result.importances_mean, index=x_llm_test.columns)
                )

    metrics = pd.DataFrame(rows)
    importances = {name: mean_importance(frames) for name, frames in importance_frames.items()}
    tuning_table = pd.concat(tuning_frames, ignore_index=True) if tuning_frames else pd.DataFrame()
    return metrics, predicted_rub, predicted_log, baseline_predictions, importances, tuning_table


def summary_table(metrics: pd.DataFrame) -> pd.DataFrame:
    """Mean/std/min/max per metric, separately for every model."""
    rows: list[dict[str, Any]] = []
    for model in MODEL_NAMES:
        subset = metrics[metrics["model"] == model]
        if subset.empty:
            continue
        for metric in METRIC_COLUMNS:
            values = subset[metric]
            rows.append({
                "model": model,
                "metric": metric,
                "mean": values.mean(),
                "std": values.std(),
                "min": values.min(),
                "max": values.max(),
            })
    return pd.DataFrame(rows)


def best_blend_weight(actual_log: np.ndarray, structured_log: np.ndarray,
                      llm_log: np.ndarray, points: int = 21) -> tuple[float, float]:
    """Grid-search the blend weight on OOF log predictions (diagnostic only).

    The weight is chosen on the same out-of-fold rows that are later reported, so the
    resulting RMSE is slightly optimistic; it is shown as a reference, not as the
    headline metric.
    """
    grid = np.linspace(0.0, 1.0, points)
    scores = [
        root_mean_squared_error(actual_log,
                                weight * structured_log + (1.0 - weight) * llm_log)
        for weight in grid
    ]
    index = int(np.argmin(scores))
    return float(grid[index]), float(scores[index])


def residual_diagnostics(predictions: pd.DataFrame, quartiles: int = 4) -> pd.DataFrame:
    """Error and bias by prediction quartile, to expose heteroscedasticity."""
    frame = predictions.copy()
    frame["prediction_group"] = pd.qcut(
        frame["predicted_rub"], quartiles,
        labels=[f"Q{index}" for index in range(1, quartiles + 1)],
    )
    return (
        frame.groupby("prediction_group", observed=True)
        .agg(
            count=("actual_rub", "size"),
            mean_predicted_rub=("predicted_rub", "mean"),
            mean_actual_rub=("actual_rub", "mean"),
            mean_residual_rub=("residual_rub", "mean"),
            mean_abs_error_rub=("abs_error_rub", "mean"),
        )
        .reset_index()
    )


def save_figures(predictions: pd.DataFrame, figure_dir: Path) -> list[Path]:
    """Out-of-fold diagnostics: predicted vs actual and residuals."""
    figure_dir.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "DejaVu Sans"],
        "axes.unicode_minus": False,
        "figure.dpi": 120,
    })
    actual = predictions["actual_rub"].to_numpy(dtype=float)
    predicted = predictions["predicted_rub"].to_numpy(dtype=float)
    outputs: list[Path] = []

    # Values and axis limits must share one unit; mixing rubles with millions clips
    # every point and leaves an empty-looking chart.
    limit = float(np.quantile(np.concatenate([actual, predicted]), 0.99)) / 1e6
    fig, ax = plt.subplots(figsize=(9, 7))
    ax.scatter(actual / 1e6, predicted / 1e6, alpha=0.4, s=18,
               color="#2563eb", edgecolors="none")
    ax.plot([0, limit], [0, limit], color="#dc2626", linewidth=1.4,
            label="идеальное совпадение")
    ax.set(title="Факт и прогноз цены, ансамбль двух HistGB (out-of-fold)",
           xlabel="Факт, млн руб.",
           ylabel="Прогноз, млн руб.", xlim=(0, limit), ylim=(0, limit))
    ax.legend()
    ax.grid(alpha=0.25)
    fig.tight_layout()
    path = figure_dir / "actual_vs_predicted.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    outputs.append(path)

    residuals = (actual - predicted) / 1e6
    cap = float(np.quantile(np.abs(residuals), 0.99))
    predicted_mln = predicted / 1e6
    x_cap = float(np.quantile(predicted_mln, 0.99))
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.scatter(predicted_mln, residuals, alpha=0.4, s=18,
               color="#7c3aed", edgecolors="none")
    ax.axhline(0, color="#dc2626", linewidth=1.4)
    ax.set(title="Остатки ансамбля двух HistGB (out-of-fold)", xlabel="Прогноз, млн руб.",
           ylabel="Остаток, млн руб.",
           xlim=(min(predicted_mln.min(), 0), x_cap), ylim=(-cap, cap))
    ax.grid(alpha=0.25)
    fig.tight_layout()
    path = figure_dir / "residuals.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    outputs.append(path)
    return outputs


def format_value(value: Any) -> str:
    """Readable value: integers as is, floats with spaces as separators."""
    if value is None or (not isinstance(value, str) and pd.isna(value)):
        return "—"
    if isinstance(value, str):
        return value
    if isinstance(value, (int, np.integer)):
        return f"{value:,}".replace(",", " ")
    if abs(float(value)) >= 1000:
        return f"{float(value):,.0f}".replace(",", " ").replace("-", "−")
    return f"{float(value):.4f}".replace(".", ",")


def markdown_table(frame: pd.DataFrame) -> str:
    """Build a small Markdown table without relying on the optional tabulate package."""
    if frame.empty:
        return "_Нет данных._"
    headers = [str(name) for name in frame.columns]
    lines = ["| " + " | ".join(headers) + " |",
             "| " + " | ".join("---" for _ in headers) + " |"]
    for row in frame.itertuples(index=False, name=None):
        lines.append("| " + " | ".join(format_value(value) for value in row) + " |")
    return "\n".join(lines)


def run(args: argparse.Namespace) -> None:
    input_path: Path = args.input
    output_dir: Path = args.output_dir
    llm_path: Path | None = args.llm_input if args.use_llm_features else None
    x, y, groups, identifiers, feature_columns = load_data(input_path, llm_path)
    structured_columns, llm_columns = split_feature_columns(feature_columns)
    use_llm = bool(llm_columns)

    if not structured_columns:
        raise SystemExit("Нет структурных признаков для первой модели HistGB.")
    if llm_path is not None and not use_llm:
        raise SystemExit(f"В {llm_path} нет колонок с префиксом '{LLM_FEATURE_PREFIX}'.")

    if args.cv == "group" and groups.nunique() < args.folds:
        raise SystemExit(
            f"Групп ({groups.nunique()}) меньше числа фолдов ({args.folds}); "
            "используйте --cv kfold или уменьшите --folds."
        )
    if len(x) < args.folds:
        raise SystemExit(f"Строк ({len(x)}) меньше числа фолдов ({args.folds}).")

    metrics, predicted_rub, predicted_log, baseline, importances, tuning = cross_validate(
        x, y, groups, structured_columns, llm_columns, args
    )
    structured_importance = importances.get(STRUCTURED_MODEL, pd.DataFrame())
    llm_importance = importances.get(LLM_MODEL, pd.DataFrame())
    summary = summary_table(metrics)

    def mean_metric(model: str, metric: str) -> float:
        subset = summary[(summary["model"] == model) & (summary["metric"] == metric)]
        return float(subset["mean"].iloc[0]) if not subset.empty else float("nan")

    # Out-of-fold rows only; a listing scored twice would distort the diagnostics.
    oof = pd.DataFrame({
        IDENTIFIER_COLUMN: identifiers.reset_index(drop=True),
        "actual_rub": y.to_numpy(dtype=float),
        "predicted_rub": predicted_rub[ENSEMBLE_MODEL],
        "predicted_structured_rub": predicted_rub[STRUCTURED_MODEL],
        "predicted_llm_rub": predicted_rub[LLM_MODEL],
        "baseline_rub": baseline,
    })
    oof["residual_rub"] = oof["actual_rub"] - oof["predicted_rub"]
    oof["abs_error_rub"] = oof["residual_rub"].abs()
    oof["predicted_log"] = predicted_log[ENSEMBLE_MODEL]

    output_dir.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(output_dir / "cv_metrics.csv", index=False, encoding="utf-8-sig")
    summary.to_csv(output_dir / "cv_summary.csv", index=False, encoding="utf-8-sig")
    oof.to_csv(output_dir / "predictions.csv", index=False, encoding="utf-8-sig")
    diagnostics = residual_diagnostics(oof)
    diagnostics.to_csv(output_dir / "residual_diagnostics.csv", index=False, encoding="utf-8-sig")
    if not structured_importance.empty:
        structured_importance.to_csv(output_dir / "feature_importance.csv", index=False,
                                     encoding="utf-8-sig")
    if not llm_importance.empty:
        llm_importance.to_csv(output_dir / "feature_importance_llm.csv", index=False,
                              encoding="utf-8-sig")
    if not tuning.empty:
        tuning.to_csv(output_dir / "tuning.csv", index=False, encoding="utf-8-sig")
    charts = save_figures(oof, output_dir / "figures")

    metric_labels = {
        "r2_log": "R² (лог)", "r2_rub": "R² (руб.)", "rmse_rub": "RMSE, ₽",
        "mae_rub": "MAE, ₽", "median_ae_rub": "MedAE, ₽", "mape_pct": "MAPE, %",
    }
    metrics_wide = (
        summary.pivot(index="metric", columns="model", values="mean")
        .reindex(METRIC_COLUMNS)
        .rename(index=metric_labels)
        .rename(columns=MODEL_LABELS)
    )
    per_fold_table = metrics[[
        "fold", "model", "n_iter", "clipped", "r2_log", "r2_rub",
        "rmse_rub", "mae_rub", "median_ae_rub", "mape_pct",
    ]].rename(columns={"fold": "Фолд", "model": "Модель", "n_iter": "Итераций",
                       "clipped": "Обрезано", **metric_labels})
    per_fold_table["Модель"] = per_fold_table["Модель"].map(MODEL_LABELS)

    model_lines = [
        f"- **{MODEL_LABELS[model]}**: R² в рублях "
        f"**{format_value(mean_metric(model, 'r2_rub'))}**, "
        f"RMSE **{format_value(mean_metric(model, 'rmse_rub'))} ₽**, "
        f"MAE **{format_value(mean_metric(model, 'mae_rub'))} ₽**."
        for model in MODEL_NAMES if (summary["model"] == model).any()
    ]

    ensemble_rmse = mean_metric(ENSEMBLE_MODEL, "rmse_rub")
    baseline_rmse = root_mean_squared_error(oof["actual_rub"], oof["baseline_rub"])
    improvement = (1 - ensemble_rmse / baseline_rmse) * 100

    blend_numbers = (
        f"Предсказания объединяются на лог-шкале: `{format_value(args.blend_weight)}·"
        f"структурная + {format_value(1 - args.blend_weight)}·LLM`."
    )
    blend_diagnostic = ""
    if use_llm:
        actual_log = np.log(oof["actual_rub"].to_numpy(dtype=float))
        chosen_rmse_log = root_mean_squared_error(
            actual_log,
            args.blend_weight * predicted_log[STRUCTURED_MODEL]
            + (1.0 - args.blend_weight) * predicted_log[LLM_MODEL],
        )
        best_weight, best_rmse_log = best_blend_weight(
            actual_log, predicted_log[STRUCTURED_MODEL], predicted_log[LLM_MODEL]
        )
        blend_numbers += (
            f" RMSE на лог-шкале при выбранном весе: **{format_value(chosen_rmse_log)}**."
        )
        blend_diagnostic = (
            f"Оптимальный по out-of-fold прогнозам вес — **{format_value(best_weight)}** "
            f"(RMSE на лог-шкале **{format_value(best_rmse_log)}**). Вес подобран на тех же "
            "OOF-прогнозах, поэтому он оптимистичен и приведён как ориентир, а не как "
            "headline-метрика."
        )
    else:
        blend_numbers = ("LLM-признаки отключены (`--no-llm-features`): ансамбль вырождается "
                         "в одну структурную модель HistGB.")

    cheapest, dearest = diagnostics.iloc[0], diagnostics.iloc[-1]
    error_gap = dearest["mean_abs_error_rub"] / cheapest["mean_abs_error_rub"]
    low_bias = float(cheapest["mean_residual_rub"])
    high_bias = float(dearest["mean_residual_rub"])
    # Describe the bias pattern from the numbers instead of asserting a sign flip:
    # depending on the target scale and clipping the ensemble may be biased one way.
    if low_bias * high_bias < 0:
        bias_sentence = (
            f"Смещение (`факт − прогноз`) меняет знак: **{format_value(low_bias)}** ₽ в нижнем "
            f"квартиле против **{format_value(high_bias)}** ₽ в верхнем, то есть ансамбль "
            "переоценивает один конец рынка и недооценивает другой."
        )
    elif low_bias >= 0 and high_bias >= 0:
        bias_sentence = (
            f"Смещение (`факт − прогноз`) остаётся положительным во всех квартилях: "
            f"**{format_value(low_bias)}** ₽ в нижнем против **{format_value(high_bias)}** ₽ "
            "в верхнем. Ансамбль систематически занижает цену, и сильнее всего — "
            "в дорогом сегменте."
        )
    else:
        bias_sentence = (
            f"Смещение (`факт − прогноз`) остаётся отрицательным во всех квартилях: "
            f"**{format_value(low_bias)}** ₽ в нижнем против **{format_value(high_bias)}** ₽ "
            "в верхнем. Ансамбль систематически завышает цену."
        )
    nan_cells = int(x.isna().sum().sum())
    rows_with_nan = int(x.isna().any(axis=1).sum())
    if args.cv == "group":
        cv_note = (
            f"- Валидация групповая (`GroupKFold` по `{GROUP_COLUMN}`): объявления одного "
            "населённого пункта не попадают одновременно в обучение и тест, поэтому оценка "
            "консервативнее обычного `KFold`. Группировка не устраняет дубликаты одного "
            "физического объекта с разными `listing_id`."
        )
    else:
        cv_note = (
            "- Валидация обычным `KFold` без группировки: объявления одного населённого "
            "пункта могут попадать и в обучение, и в тест, что завышает метрики. "
            "Для честной оценки используйте `--cv group`."
        )

    report = [
        "# Ансамбль из двух HistGB: структурные признаки и LLM-признаки",
        "",
        f"Данные: `{input_path}`"
        + (f" + LLM-признаки `{llm_path}`" if use_llm
           else " (LLM-признаки отключены)"),
        "",
        "## Состав ансамбля",
        f"- **{MODEL_LABELS[STRUCTURED_MODEL]}**: `HistGradientBoostingRegressor` "
        f"(`max_iter={args.max_iter}`, early stopping; признаков: {len(structured_columns)}).",
        *([f"- **{MODEL_LABELS[LLM_MODEL]}**: `HistGradientBoostingRegressor` "
           f"(`max_iter={args.max_iter}`, early stopping; LLM-признаков: {len(llm_columns)})."]
          if use_llm else ["- Модель на LLM-признаках отключена (`--no-llm-features`)."]),
        f"- **{MODEL_LABELS[ENSEMBLE_MODEL]}**: взвешенное среднее на лог-шкале.",
        "",
        f"Признаков всего: **{len(feature_columns)}** (структурных: "
        f"**{len(structured_columns)}**, LLM: **{len(llm_columns)}**); "
        f"строк: **{len(x):,}**; пропусков в матрице: **{nan_cells:,}**.",
        f"Цель: `log({TARGET_COLUMN})`; метрики пересчитаны в рубли через `expm1`.",
        f"Валидация: {'GroupKFold по `' + GROUP_COLUMN + '`' if args.cv == 'group' else 'KFold'} "
        f"({args.folds} фолдов, групп: {groups.nunique():,}, seed {args.seed}); оба члена "
        "ансамбля обучаются на одних и тех же фолдах.",
        f"Подбор гиперпараметров структурной модели: "
        f"{'внутренний перебор на ' + str(args.tune) + ' кандидатах' if args.tune else 'не выполнялся'}.",
        "",
        "## Метрики (среднее по фолдам)",
        "",
        *model_lines,
        "",
        "### Средние значения по метрикам",
        markdown_table(metrics_wide.reset_index().rename(columns={"metric": "Метрика"})),
        "",
        "## Метрики по фолдам",
        markdown_table(per_fold_table),
        "",
        "## Смешивание моделей",
        blend_numbers,
        *([blend_diagnostic] if blend_diagnostic else []),
        "",
        "## Сравнение с базовой моделью",
        "Базовая модель предсказывает среднее по обучающей части фолда "
        "(обратное преобразование лога — геометрическое среднее): "
        f"RMSE **{format_value(baseline_rmse)}** ₽.",
        f"Ансамбль снижает RMSE относительно базовой модели на **{improvement:.1f} %**.",
        "",
    ]

    if not structured_importance.empty:
        report += [
            "## Важность признаков структурной модели",
            "Permutation importance на тестовых фолдах (без утечки из обучения). "
            "Падение R² на лог-шкале:",
            markdown_table(structured_importance.head(15).rename(columns={
                "feature": "Признак", "importance_mean_r2_log": "Падение R²",
            })),
            "",
        ]

    if not llm_importance.empty:
        report += [
            "## Важность LLM-признаков",
            "Permutation importance второй модели (обучена только на `llm_*`), "
            "на тестовых фолдах:",
            markdown_table(llm_importance.head(15).rename(columns={
                "feature": "Признак", "importance_mean_r2_log": "Падение R²",
            })),
            "",
        ]

    if not tuning.empty:
        best_rows = tuning.sort_values("rmse_log_mean").groupby("fold").head(1)
        report += [
            "## Подбор гиперпараметров структурной модели",
            "Поиск выполняется на внутренней групповой разбивке обучающей части фолда, "
            "тестовый фолд в подборе не участвует. Лучший кандидат каждого фолда:",
            markdown_table(best_rows[[
                "fold", "rmse_log_mean", "learning_rate", "max_leaf_nodes",
                "min_samples_leaf", "l2_regularization",
            ]].rename(columns={
                "fold": "Фолд", "rmse_log_mean": "RMSE (лог)",
            })),
            "",
        ]

    report += [
        "## Диагностика остатков ансамбля",
        "Средняя абсолютная ошибка и смещение по квартилям прогноза:",
        markdown_table(diagnostics.rename(columns={
            "prediction_group": "Квартиль прогноза",
            "count": "Строк",
            "mean_predicted_rub": "Средний прогноз, ₽",
            "mean_actual_rub": "Средний факт, ₽",
            "mean_residual_rub": "Среднее смещение, ₽",
            "mean_abs_error_rub": "Средняя абс. ошибка, ₽",
        })),
        f"Средняя абсолютная ошибка растёт от нижнего квартиля прогноза "
        f"(**{format_value(cheapest['mean_abs_error_rub'])}** ₽) к верхнему "
        f"(**{format_value(dearest['mean_abs_error_rub'])}** ₽) — в **{error_gap:.1f}** раза.",
        bias_sentence,
        "Остаточная гетероскедастичность ожидаема: дорогие объекты разнороднее, "
        "а обучающих примеров в верхнем сегменте мало.",
        "",
        "## Обработка пропусков",
        "Пропуски **не заполняются**: обе модели — `HistGradientBoostingRegressor`, "
        "который сам направляет объекты с NaN в отдельную ветвь. Это принципиально, "
        f"потому что в матрице **{nan_cells:,}** пропущенных ячеек, и хотя бы один "
        f"пропуск есть в **{rows_with_nan:,}** из **{len(x):,}** строк "
        f"({rows_with_nan / len(x) * 100:.1f} %). Построчное удаление пропусков "
        f"оставило бы **{len(x) - rows_with_nan:,}** строк — HistGB сохраняет все данные.",
        "",
        "## Ограничения",
        "- `asking_price_rub` — цена предложения, а не подтверждённая цена сделки.",
        cv_note,
        "- Вес ансамбля по умолчанию фиксирован (0,5 на лог-шкале). Оптимальный вес из "
        "отчёта подобран на out-of-fold прогнозах и потому оптимистичен: для честной оценки "
        "подбирайте вес внутри обучающих фолдов или на отдельном наборе.",
        *(["- LLM-признаки извлечены языковой моделью из текста объявления: они шумные, "
           "могут содержать пропуски и ошибки распознавания. Модель на LLM-признаках "
           "обучена только на них, поэтому не видит структурных характеристик объекта."]
          if use_llm else []),
        "- Целевая переменная скошена (skew ≈ 9,8): RMSE сильно зависит от нескольких "
        "дорогих объектов, поэтому рядом приведены MAE и медианная ошибка.",
        "- Признак `badge_mortgage_discount_pct` — прокси продавца/сегмента "
        "(корреляция с ценой 0,82, немонотонен по цене). Проверяйте устойчивость "
        "результата без группы `badge_*`.",
        "- Прогнозы обрезаются по диапазону обучающей части фолда "
        f"({'включено' if args.clip_predictions else 'выключено'}), "
        "поэтому модель не выдаёт цены за пределами наблюдённого диапазона.",
        "- Ансамбль не экстраполирует: для объектов вне обучающего распределения "
        "(другой регион, другой тип недвижимости) прогноз не обоснован.",
        "",
        "## Файлы",
        f"- Метрики по фолдам: `{output_dir / 'cv_metrics.csv'}`",
        f"- Сводка метрик: `{output_dir / 'cv_summary.csv'}`",
        f"- Прогнозы out-of-fold (ансамбль, структурная, LLM): "
        f"`{output_dir / 'predictions.csv'}`",
        f"- Диагностика остатков: `{output_dir / 'residual_diagnostics.csv'}`",
        *([f"- Важность признаков структурной модели: "
           f"`{output_dir / 'feature_importance.csv'}`"]
          if not structured_importance.empty else []),
        *([f"- Важность LLM-признаков: `{output_dir / 'feature_importance_llm.csv'}`"]
          if not llm_importance.empty else []),
        *([f"- Результаты подбора: `{output_dir / 'tuning.csv'}`"] if not tuning.empty else []),
        *[f"- График: `{path}`" for path in charts],
    ]
    (output_dir / "report.md").write_text("\n".join(report) + "\n", encoding="utf-8")

    print(f"Rows: {len(x):,}; structured features: {len(structured_columns)}; "
          f"LLM features: {len(llm_columns)}; folds: {args.folds}".replace(",", " "))
    for model in MODEL_NAMES:
        if (summary["model"] == model).any():
            r2 = mean_metric(model, "r2_rub")
            mae = mean_metric(model, "mae_rub")
            print(f"{MODEL_LABELS[model]}: R2 (rub) = {r2:.3f}; "
                  f"MAE (rub) = {mae:,.0f}".replace(",", " "))
    print(f"Baseline RMSE (rub): {baseline_rmse:,.0f}".replace(",", " "))
    print(f"Reports: {output_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Ансамбль из двух HistGradientBoosting: структурные признаки и "
                    "LLM-признаки, с кросс-валидацией."
    )
    parser.add_argument(
        "--input", type=Path, default=default_input_path(),
        help="Матрица структурных признаков (по умолчанию data/features/features_rich.csv).",
    )
    parser.add_argument(
        "--llm-input", type=Path, default=default_llm_input_path(),
        help="Файл LLM-признаков, объединяется по listing_id "
             "(по умолчанию data/features/features_llm.csv).",
    )
    parser.add_argument(
        "--no-llm-features", dest="use_llm_features", action="store_false", default=True,
        help="Отключить вторую модель на LLM-признаках (останется только структурная).",
    )
    parser.add_argument(
        "--blend-weight", type=float, default=0.5,
        help="Вес структурной модели в ансамбле на лог-шкале (0..1; LLM-модель получает 1-w).",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=default_report_dir(),
        help="Каталог для отчёта и метрик (по умолчанию reports/histgb_reach+llm).",
    )
    parser.add_argument(
        "--cv", choices=("group", "kfold"), default="group",
        help="'group' — GroupKFold по населённому пункту (честно), 'kfold' — быстрее, но оптимистично.",
    )
    parser.add_argument("--folds", type=int, default=5, help="Число фолдов (по умолчанию 5).")
    parser.add_argument("--seed", type=int, default=42, help="Случайное зерно.")
    parser.add_argument("--max-iter", type=int, default=400, help="Максимум итераций бустинга.")
    parser.add_argument("--tune", type=int, default=0,
                        help="Число случайных кандидатов гиперпараметров структурной модели "
                             "на фолд (0 — без подбора).")
    parser.add_argument("--inner-folds", type=int, default=3,
                        help="Фолдов во внутренней разбивке при подборе гиперпараметров.")
    parser.add_argument("--importance-repeats", type=int, default=5,
                        help="Повторов permutation importance для каждой модели (0 — не считать).")
    parser.add_argument("--no-clip-predictions", dest="clip_predictions", action="store_false",
                        default=True,
                        help="Отключить обрезку прогнозов по диапазону обучающей части фолда.")
    parser.add_argument("--verbose", action="store_true",
                        help="Показывать предупреждения и подробные сообщения.")
    args = parser.parse_args()
    if not 0.0 <= args.blend_weight <= 1.0:
        parser.error("--blend-weight должен быть в диапазоне 0..1.")
    return args


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(levelname)s: %(message)s",
    )
    run(args)


if __name__ == "__main__":
    main()
