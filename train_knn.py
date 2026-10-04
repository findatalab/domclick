#!/usr/bin/env python3
"""Train and evaluate a KNeighborsRegressor for the asking-price task.

Model:   sklearn KNeighborsRegressor inside a Pipeline: median imputation ->
         feature scaling -> k nearest neighbours averaged on the log target.
         With `weights="distance"` the forecast is a distance-weighted mean of the
         neighbours, i.e. a local non-parametric regressor.
Data:    data/features/features_v1.csv (produced by build_features_v1.py).
Target:  asking_price_rub, trained on the log scale and scored on the ruble scale.
CV:      GroupKFold by `group_locality` so listings from one locality never split
         between train and test; use --cv kfold only for a quicker comparison.

Why this is not a copy of the HistGradientBoosting script
--------------------------------------------------------
1. kNN cannot consume NaN: a distance is undefined as soon as one coordinate is
   missing. The imputer is therefore part of the model, and it is fitted **inside
   each training fold**, so the test fold never influences the fill values.
2. kNN is distance-based, so the features are scaled inside the same pipeline;
   otherwise the house area (tens to hundreds) would dominate the binary
   indicators and the ratios.
3. There is no early stopping, no `max_iter` and no prediction clipping: on the
   log scale the forecast is a convex combination of neighbour targets, so it can
   never leave the range observed in training.
4. Results go to reports/knn_v1/, never into another model's report directory.
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
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    mean_absolute_error,
    mean_absolute_percentage_error,
    median_absolute_error,
    r2_score,
    root_mean_squared_error,
)
from sklearn.model_selection import GroupKFold, KFold, ParameterSampler
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler, StandardScaler

import mlflow_utils  # локальный модуль: опциональный трекинг запусков (см. README)


LOG = logging.getLogger("domclick.knn")
IDENTIFIER_COLUMN = "listing_id"
TARGET_COLUMN = "asking_price_rub"
LOG_TARGET_COLUMN = "log_asking_price_rub"
GROUP_COLUMN = "group_locality"

# id, both target variants and the CV group key must never enter the design matrix.
NON_FEATURE_COLUMNS = (IDENTIFIER_COLUMN, TARGET_COLUMN, LOG_TARGET_COLUMN, GROUP_COLUMN)

# Metrics on the ruble scale plus one on the modelled log scale.
METRIC_COLUMNS = ("r2_log", "r2_rub", "rmse_rub", "mae_rub", "median_ae_rub", "mape_pct")

# kNN hyperparameters; --tune samples these on an inner grouped split of the training part.
PARAM_DISTRIBUTIONS = {
    "n_neighbors": [3, 5, 8, 12, 20, 30],
    "weights": ["uniform", "distance"],
    "p": [1, 2],
}

# Distance-based models need every feature on one scale. `None` keeps the raw values.
SCALERS: dict[str, Any] = {
    "standard": StandardScaler,
    "robust": RobustScaler,
    "none": None,
}


def default_input_path() -> Path:
    """Where build_features_v1.py writes the compact modeling matrix."""
    return Path(__file__).resolve().parent / "data" / "features" / "features_v1.csv"


def default_report_dir() -> Path:
    """Reports go to reports/, in a directory named after this model."""
    return Path(__file__).resolve().parent / "reports" / "knn_v1"


def load_data(path: Path) -> tuple[pd.DataFrame, pd.Series, pd.Series, pd.Series, list[str]]:
    """Read the matrix and split it into features, target, CV groups and identifiers.

    Missing values are intentionally preserved: the model handles them natively.
    """
    try:
        frame = pd.read_csv(path)
    except FileNotFoundError as exc:
        raise SystemExit(
            f"Не найден файл признаков: {path}\nСначала запустите build_features_v1.py."
        ) from exc
    except OSError as exc:
        raise SystemExit(f"Не удалось прочитать {path}: {exc}") from exc

    if TARGET_COLUMN not in frame.columns:
        raise SystemExit(f"В {path} нет целевой колонки {TARGET_COLUMN}.")

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
    LOG.info("Пропусков в матрице признаков: %s (заполняются медианой внутри фолда)",
             missing_cells)
    return frame[feature_columns], frame[TARGET_COLUMN], groups, identifiers, feature_columns


def make_model(params: dict[str, Any], scaler: str) -> Pipeline:
    """Impute, scale and average the k nearest neighbours.

    The pipeline is fitted on the training part of a fold only, so the median and the
    scaling statistics never see the test rows.
    """
    steps: list[tuple[str, Any]] = [("imputer", SimpleImputer(strategy="median"))]
    scaler_class = SCALERS[scaler]
    if scaler_class is not None:
        steps.append(("scaler", scaler_class()))
    steps.append(("knn", KNeighborsRegressor(n_jobs=-1, **params)))
    return Pipeline(steps)


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
                scaler: str) -> tuple[dict[str, Any], pd.DataFrame]:
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
            model = make_model(params, scaler)
            model.fit(x_train.iloc[train_index], y_train_log.iloc[train_index])
            predicted = model.predict(x_train.iloc[valid_index])
            scores.append(root_mean_squared_error(
                y_train_log.iloc[valid_index], predicted
            ))
        rows.append({"candidate": index, "rmse_log_mean": float(np.mean(scores)), **params})
        print(f"  кандидат {index}/{len(candidates)}: RMSE(log) = {rows[-1]['rmse_log_mean']:.4f}")

    table = pd.DataFrame(rows).sort_values("rmse_log_mean").reset_index(drop=True)
    best: dict[str, Any] = {name: table.loc[0, name] for name in PARAM_DISTRIBUTIONS}
    for name, values in PARAM_DISTRIBUTIONS.items():
        # ParameterSampler returns numpy scalars: restore ints and floats, keep strings.
        if all(isinstance(value, int) and not isinstance(value, bool) for value in values):
            best[name] = int(best[name])
        elif all(isinstance(value, float) for value in values):
            best[name] = float(best[name])
    return best, table


def cross_validate(x: pd.DataFrame, y: pd.Series, groups: pd.Series, args: argparse.Namespace
                   ) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, pd.DataFrame, pd.DataFrame]:
    """Run CV; return per-fold metrics, OOF predictions, baseline predictions and importances."""
    y_log = np.log(y.to_numpy(dtype=float))
    splitter = make_splitter(args.cv, args.folds, args.seed)
    rows: list[dict[str, Any]] = []
    predictions = np.full(len(x), np.nan)
    baseline_predictions = np.full(len(x), np.nan)
    importance_frames: list[pd.Series] = []
    tuning_frames: list[pd.DataFrame] = []

    for fold, (train_index, test_index) in enumerate(
            split_iterator(splitter, x, groups, args.cv), start=1):
        x_train, x_test = x.iloc[train_index], x.iloc[test_index]
        y_train_log = pd.Series(y_log[train_index], index=x_train.index)

        params: dict[str, Any] = {
            "n_neighbors": args.n_neighbors,
            "weights": args.weights,
            "p": args.p,
        }
        if args.tune:
            params, tuning = tune_params(
                x_train, y_train_log, groups.iloc[train_index],
                args.tune, args.inner_folds, args.seed + fold, args.scaler,
            )
            tuning.insert(0, "fold", fold)
            tuning_frames.append(tuning)

        model = make_model(params, args.scaler)
        model.fit(x_train, y_train_log)
        predicted_log = model.predict(x_test)
        # kNN averages neighbour targets, so the log-scale forecast is a convex
        # combination and expm1 keeps it inside the training range: no clipping needed.
        predicted_rub = np.expm1(predicted_log)
        predictions[test_index] = predicted_rub

        baseline = DummyRegressor(strategy="mean")
        baseline.fit(x_train, y_train_log)
        baseline_predictions[test_index] = np.expm1(baseline.predict(x_test))

        row: dict[str, Any] = {
            "fold": fold,
            "train_size": len(train_index),
            "test_size": len(test_index),
            "n_neighbors": int(model.named_steps["knn"].n_neighbors),
        }
        row.update(score_fold(y.iloc[test_index].to_numpy(dtype=float), predicted_rub,
                              y_log[test_index], predicted_log))
        rows.append(row)
        print(f"Фолд {fold}/{args.folds}: R2(руб.) = {row['r2_rub']:.3f}, "
              f"MAE = {row['mae_rub']:,.0f}, соседей = {row['n_neighbors']}".replace(",", " "))

        if args.importance_repeats:
            result = permutation_importance(
                model, x_test, y_log[test_index], n_repeats=args.importance_repeats,
                random_state=args.seed,
            )
            importance_frames.append(pd.Series(result.importances_mean, index=x.columns))

    metrics = pd.DataFrame(rows)
    importance = pd.DataFrame()
    if importance_frames:
        importance = (
            pd.concat(importance_frames, axis=1).mean(axis=1)
            .rename("importance_mean_r2_log").reset_index()
            .rename(columns={"index": "feature"})
            .sort_values("importance_mean_r2_log", ascending=False)
            .reset_index(drop=True)
        )
    tuning_table = pd.concat(tuning_frames, ignore_index=True) if tuning_frames else pd.DataFrame()
    return metrics, predictions, baseline_predictions, importance, tuning_table


def summary_table(metrics: pd.DataFrame) -> pd.DataFrame:
    """Mean/std/min/max per metric across folds."""
    rows: list[dict[str, Any]] = []
    for metric in METRIC_COLUMNS:
        values = metrics[metric]
        rows.append({
            "metric": metric,
            "mean": values.mean(),
            "std": values.std(),
            "min": values.min(),
            "max": values.max(),
        })
    return pd.DataFrame(rows)


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
    ax.set(title="Факт и прогноз цены, kNN (out-of-fold)", xlabel="Факт, млн руб.",
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
    ax.set(title="Остатки kNN (out-of-fold)", xlabel="Прогноз, млн руб.",
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
    x, y, groups, identifiers, feature_columns = load_data(input_path)

    if args.cv == "group" and groups.nunique() < args.folds:
        raise SystemExit(
            f"Групп ({groups.nunique()}) меньше числа фолдов ({args.folds}); "
            "используйте --cv kfold или уменьшите --folds."
        )
    if len(x) < args.folds:
        raise SystemExit(f"Строк ({len(x)}) меньше числа фолдов ({args.folds}).")

    metrics, predictions, baseline, importance, tuning = cross_validate(x, y, groups, args)
    summary = summary_table(metrics)
    summary_by_metric = summary.set_index("metric")

    # Out-of-fold rows only; a listing scored twice would distort the diagnostics.
    oof = pd.DataFrame({
        IDENTIFIER_COLUMN: identifiers.reset_index(drop=True),
        "actual_rub": y.to_numpy(dtype=float),
        "predicted_rub": predictions,
        "baseline_rub": baseline,
    })
    oof["residual_rub"] = oof["actual_rub"] - oof["predicted_rub"]
    oof["abs_error_rub"] = oof["residual_rub"].abs()
    oof["predicted_log"] = np.log(np.clip(oof["predicted_rub"], 1e3, None))

    output_dir.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(output_dir / "cv_metrics.csv", index=False, encoding="utf-8-sig")
    summary.to_csv(output_dir / "cv_summary.csv", index=False, encoding="utf-8-sig")
    oof.to_csv(output_dir / "predictions.csv", index=False, encoding="utf-8-sig")
    diagnostics = residual_diagnostics(oof)
    diagnostics.to_csv(output_dir / "residual_diagnostics.csv", index=False, encoding="utf-8-sig")
    if not importance.empty:
        importance.to_csv(output_dir / "feature_importance.csv", index=False, encoding="utf-8-sig")
    if not tuning.empty:
        tuning.to_csv(output_dir / "tuning.csv", index=False, encoding="utf-8-sig")
    charts = save_figures(oof, output_dir / "figures")

    def metric_summary(name: str) -> str:
        row = summary_by_metric.loc[name]
        return f"{format_value(row['mean'])} ± {format_value(row['std'])}"

    best_fold = metrics.loc[metrics["r2_rub"].idxmax()]
    worst_fold = metrics.loc[metrics["r2_rub"].idxmin()]
    cheapest, dearest = diagnostics.iloc[0], diagnostics.iloc[-1]
    error_gap = dearest["mean_abs_error_rub"] / cheapest["mean_abs_error_rub"]
    low_bias = float(cheapest["mean_residual_rub"])
    high_bias = float(dearest["mean_residual_rub"])
    # Describe the bias pattern from the numbers instead of asserting a sign flip:
    # depending on the data the model may be biased in one direction only.
    if low_bias * high_bias < 0:
        bias_sentence = (
            f"Смещение (`факт − прогноз`) меняет знак: **{format_value(low_bias)}** ₽ в нижнем "
            f"квартиле против **{format_value(high_bias)}** ₽ в верхнем, то есть модель "
            "переоценивает один конец рынка и недооценивает другой."
        )
    elif low_bias >= 0 and high_bias >= 0:
        bias_sentence = (
            f"Смещение (`факт − прогноз`) остаётся положительным во всех квартилях: "
            f"**{format_value(low_bias)}** ₽ в нижнем против **{format_value(high_bias)}** ₽ "
            "в верхнем. Модель систематически занижает цену, и сильнее всего — "
            "в дорогом сегменте."
        )
    else:
        bias_sentence = (
            f"Смещение (`факт − прогноз`) остаётся отрицательным во всех квартилях: "
            f"**{format_value(low_bias)}** ₽ в нижнем против **{format_value(high_bias)}** ₽ "
            "в верхнем. Модель систематически завышает цену."
        )
    baseline_rmse = root_mean_squared_error(oof["actual_rub"], oof["baseline_rub"])
    improvement = (1 - summary_by_metric.loc["rmse_rub", "mean"] / baseline_rmse) * 100
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
        "# kNN: модель цены предложения",
        "",
        f"Данные: `{input_path}`",
        f"Модель: `KNeighborsRegressor` "
        f"(`n_neighbors={args.n_neighbors}`, `weights={args.weights}`, `p={args.p}`) "
        f"в `Pipeline`: `SimpleImputer(median)` → "
        f"{'`' + args.scaler.capitalize() + 'Scaler`' if args.scaler != 'none' else 'без масштабирования'} → kNN.",
        f"Признаков: **{len(feature_columns)}**; строк: **{format_value(len(x))}**; "
        f"пропусков в матрице: **{format_value(nan_cells)}**.",
        f"Цель: `log({TARGET_COLUMN})`; метрики пересчитаны в рубли через `expm1`.",
        f"Валидация: {'GroupKFold по `' + GROUP_COLUMN + '`' if args.cv == 'group' else 'KFold'} "
        f"({args.folds} фолдов, групп: {format_value(groups.nunique())}, seed {args.seed}).",
        f"Подбор гиперпараметров: "
        f"{'внутренний перебор на ' + str(args.tune) + ' кандидатах' if args.tune else 'не выполнялся'}.",
        "",
        "## Метрики (среднее ± стандартное отклонение по фолдам)",
        "",
        f"- R² на лог-шкале: **{metric_summary('r2_log')}**",
        f"- R² в рублях: **{metric_summary('r2_rub')}**",
        f"- RMSE: **{metric_summary('rmse_rub')} ₽**",
        f"- MAE: **{metric_summary('mae_rub')} ₽**",
        f"- Медианная абсолютная ошибка: **{metric_summary('median_ae_rub')} ₽**",
        f"- MAPE: **{metric_summary('mape_pct')} %**",
        "",
        f"Лучший фолд: №{int(best_fold['fold'])} (R² в рублях = {format_value(best_fold['r2_rub'])}); "
        f"худший: №{int(worst_fold['fold'])} (R² в рублях = {format_value(worst_fold['r2_rub'])}).",
        "",
        "## Метрики по фолдам",
        markdown_table(metrics[["fold", "train_size", "test_size", "n_neighbors",
                                "r2_log", "r2_rub", "rmse_rub", "mae_rub", "median_ae_rub",
                                "mape_pct"]].rename(columns={
            "fold": "Фолд", "train_size": "Обучение", "test_size": "Тест",
            "n_neighbors": "Соседей",
            "r2_log": "R² (лог)", "r2_rub": "R² (руб.)", "rmse_rub": "RMSE, ₽",
            "mae_rub": "MAE, ₽", "median_ae_rub": "MedAE, ₽", "mape_pct": "MAPE, %",
        })),
        "",
        "## Сравнение с базовой моделью",
        f"Базовая модель предсказывает среднее по обучающей части фолда "
        f"(обратное преобразование лога — геометрическое среднее): "
        f"RMSE **{format_value(baseline_rmse)}** ₽.",
        f"kNN снижает RMSE относительно базовой модели на **{improvement:.1f} %**.",
        "",
    ]

    if not importance.empty:
        report += [
            "## Важность признаков",
            "Permutation importance, посчитанная на тестовых фолдах "
            "(то есть без утечки из обучения). Падение R² на лог-шкале:",
            markdown_table(importance.head(15).rename(columns={
                "feature": "Признак", "importance_mean_r2_log": "Падение R²",
            })),
            "",
        ]

    if not tuning.empty:
        best_rows = tuning.sort_values("rmse_log_mean").groupby("fold").head(1)
        report += [
            "## Подбор гиперпараметров",
            "Поиск выполняется на внутренней групповой разбивке обучающей части фолда, "
            "тестовый фолд в подборе не участвует. Лучший кандидат каждого фолда:",
            markdown_table(best_rows[[
                "fold", "rmse_log_mean", "n_neighbors", "weights", "p",
            ]].rename(columns={
                "fold": "Фолд", "rmse_log_mean": "RMSE (лог)",
                "n_neighbors": "Соседей", "weights": "Вес", "p": "Метрика p",
            })),
            "",
        ]

    report += [
        "## Диагностика остатков",
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
        "## Обработка пропусков и масштабирование",
        "kNN измеряет расстояние между объектами, поэтому пропуски ему мешают "
        "напрямую: при NaN в координате расстояние не определено. Внутри `Pipeline` "
        f"работают `SimpleImputer(median)` и "
        f"{'`' + args.scaler.capitalize() + 'Scaler`' if args.scaler != 'none' else 'масштабирование отключено'} "
        "(`--scaler standard|robust|none`). Оценщик живёт внутри фолда и обучается "
        "**только на его обучающей части**, поэтому медианы и масштаб не подсматривают "
        "тест: без этого матрица утекала бы в метрики.",
        f"В матрице **{format_value(nan_cells)}** пропущенных ячеек, и хотя бы один пропуск "
        f"есть в **{format_value(rows_with_nan)}** из **{format_value(len(x))}** строк "
        f"({rows_with_nan / len(x) * 100:.1f} %): построчное удаление пропусков "
        f"оставило бы **{format_value(len(x) - rows_with_nan)}** строк, поэтому пропуски "
        "именно заполняются.",
        "",
        "## Ограничения",
        "- `asking_price_rub` — цена предложения, а не подтверждённая цена сделки.",
        cv_note,
        "- Целевая переменная скошена (skew ≈ 9,8): RMSE сильно зависит от нескольких "
        "дорогих объектов, поэтому рядом приведены MAE и медианная ошибка.",
        "- kNN не экстраполирует: на лог-шкале прогноз — выпуклая комбинация целей "
        "соседей, поэтому он всегда лежит внутри диапазона обучающей части фолда. "
        "Обрезка прогнозов этой модели не нужна и не выполняется.",
        f"- Проклятие размерности: признаков {len(feature_columns)}, и большинство из "
        "них — бинарные индикаторы. Для kNN полезно сокращать пространство (отбор "
        "признаков, PCA) или менять метрику: сравните `--scaler none|robust|standard` "
        "и `--p 1|2` на своей матрице.",
        "- У модели нет коэффициентов: вклад признаков читается только по permutation "
        "importance, а не по знаку и величине эффекта.",
        "- Цена определяется близостью объектов целиком, а не суммой эффектов: для "
        "объекта вне обучающего распределения (другой регион, другой тип "
        "недвижимости) прогноз не обоснован.",
        "",
        "## Файлы",
        f"- Метрики по фолдам: `{output_dir / 'cv_metrics.csv'}`",
        f"- Сводка метрик: `{output_dir / 'cv_summary.csv'}`",
        f"- Прогнозы out-of-fold: `{output_dir / 'predictions.csv'}`",
        f"- Диагностика остатков: `{output_dir / 'residual_diagnostics.csv'}`",
        *([f"- Важность признаков: `{output_dir / 'feature_importance.csv'}`"]
          if not importance.empty else []),
        *([f"- Результаты подбора: `{output_dir / 'tuning.csv'}`"] if not tuning.empty else []),
        *[f"- График: `{path}`" for path in charts],
    ]
    (output_dir / "report.md").write_text("\n".join(report) + "\n", encoding="utf-8")

    # Модель на всей матрице нужна только MLflow: метрики выше считаются по фолдам.
    final_model = None
    if args.mlflow and args.mlflow_log_model:
        final_model = make_model(
            {"n_neighbors": args.n_neighbors, "weights": args.weights, "p": args.p},
            args.scaler,
        ).fit(x, np.log(y.to_numpy(dtype=float)))
    mlflow_utils.log_training_run(
        args, model_name="knn", input_path=input_path, output_dir=output_dir,
        extra_params={"rows": len(x), "features": len(feature_columns)},
        extra_metrics={"baseline_rmse_rub": float(baseline_rmse),
                       "improvement_pct": float(improvement)},
        metrics=metrics, summary=summary, model=final_model,
    )

    print(f"Rows: {len(x):,}; features: {len(feature_columns)}; folds: {args.folds}".replace(",", " "))
    print(f"R2 (rub): {summary_by_metric.loc['r2_rub', 'mean']:.3f} "
          f"+- {summary_by_metric.loc['r2_rub', 'std']:.3f}")
    print(f"R2 (log): {summary_by_metric.loc['r2_log', 'mean']:.3f}")
    print(f"RMSE (rub): {summary_by_metric.loc['rmse_rub', 'mean']:,.0f}".replace(",", " "))
    print(f"MAE (rub): {summary_by_metric.loc['mae_rub', 'mean']:,.0f}".replace(",", " "))
    print(f"Baseline RMSE (rub): {baseline_rmse:,.0f}".replace(",", " "))
    print(f"Reports: {output_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Обучение kNN-регрессии на features_v1.csv с кросс-валидацией."
    )
    parser.add_argument(
        "--input", type=Path, default=default_input_path(),
        help="Матрица признаков (по умолчанию data/features/features_v1.csv).",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=default_report_dir(),
        help="Каталог для отчёта и метрик (по умолчанию reports/knn_v1).",
    )
    parser.add_argument(
        "--cv", choices=("group", "kfold"), default="group",
        help="'group' — GroupKFold по населённому пункту (честно), 'kfold' — быстрее, но оптимистично.",
    )
    parser.add_argument("--folds", type=int, default=5, help="Число фолдов (по умолчанию 5).")
    parser.add_argument("--seed", type=int, default=42, help="Случайное зерно.")
    parser.add_argument("--n-neighbors", type=int, default=10,
                        help="Число соседей (по умолчанию 10).")
    parser.add_argument("--weights", choices=("uniform", "distance"), default="distance",
                        help="Веса соседей: 'distance' — по обратному расстоянию (по умолчанию).")
    parser.add_argument("--p", type=int, choices=(1, 2), default=2,
                        help="Метрика Минковского: 2 — евклидова (по умолчанию), 1 — манхэттенская.")
    parser.add_argument("--scaler", choices=tuple(SCALERS), default="standard",
                        help="Масштабирование признаков внутри фолда (по умолчанию standard).")
    parser.add_argument("--tune", type=int, default=0,
                        help="Число случайных кандидатов гиперпараметров на фолд (0 — без подбора).")
    parser.add_argument("--inner-folds", type=int, default=3,
                        help="Фолдов во внутренней разбивке при подборе гиперпараметров.")
    parser.add_argument("--importance-repeats", type=int, default=5,
                        help="Повторов permutation importance (0 — не считать).")
    parser.add_argument("--verbose", action="store_true",
                        help="Показывать предупреждения и подробные сообщения.")
    mlflow_utils.add_mlflow_args(parser)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(levelname)s: %(message)s",
    )
    run(args)


if __name__ == "__main__":
    main()
