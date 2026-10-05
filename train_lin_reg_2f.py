#!/usr/bin/env python3
"""Train a simple linear regression (scikit-learn) and evaluate it with k-fold CV.

Model:    LinearRegression on the numeric feature columns of the input table.
Target:   asking_price_rub (asking price, not a confirmed sale price).
Default:  5-fold KFold cross-validation.

Results follow the shared experiment contract (see docs/experiment_contract.md):
each run gets its own numbered directory under `experiments/` (`01`, `02`, ...) with
`meta.json`, `report.md`, canonical `metrics.csv`, `metrics_summary.csv`,
`predictions.csv`, residual diagnostics and figures. The run is registered in
`experiments/index.csv`. Report plumbing lives in `report_utils.py`.

The feature set is derived from the input table (every column except the
identifier and the target), so adding features in build_features.py is enough.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    mean_absolute_error,
    mean_absolute_percentage_error,
    median_absolute_error,
    r2_score,
    root_mean_squared_error,
)
from sklearn.model_selection import KFold

import report_utils


LOG = logging.getLogger("domclick.linreg")
IDENTIFIER_COLUMNS = ("listing_id",)
TARGET_COLUMN = "asking_price_rub"
MODEL_NAME = "lin-reg-2f"
# Canonical metric names required by the experiment contract (r2 in ruble terms).
METRIC_COLUMNS = ("r2_rub", "rmse_rub", "mae_rub", "median_ae_rub", "mape_pct")


def default_input_path() -> Path:
    """Where build_features.py writes the modeling table."""
    return Path(__file__).resolve().parent / "data" / "features" / "features.csv"


def load_features(path: Path) -> tuple[pd.DataFrame, pd.Series]:
    """Read the feature table and return (features, target) with aligned indices."""
    try:
        frame = pd.read_csv(path)
    except FileNotFoundError as exc:
        raise SystemExit(
            f"Не найден файл признаков: {path}\nСначала запустите build_features.py."
        ) from exc
    except OSError as exc:
        raise SystemExit(f"Не удалось прочитать {path}: {exc}") from exc

    if TARGET_COLUMN not in frame.columns:
        raise SystemExit(f"В {path} нет целевой колонки {TARGET_COLUMN}.")

    feature_columns = [
        column for column in frame.columns
        if column not in (*IDENTIFIER_COLUMNS, TARGET_COLUMN)
    ]
    if not feature_columns:
        raise SystemExit(f"В {path} нет ни одного признака кроме идентификатора и цели.")

    frame = frame[[*feature_columns, TARGET_COLUMN]].copy()
    for column in (*feature_columns, TARGET_COLUMN):
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
        frame[column] = frame[column].replace([np.inf, -np.inf], np.nan)

    incomplete = frame.isna().any(axis=1)
    if incomplete.any():
        LOG.warning("Строк с пропусками пропущено: %s", int(incomplete.sum()))
        frame = frame.loc[~incomplete]
    if frame.empty:
        raise SystemExit("После удаления пропусков не осталось строк.")

    return frame[feature_columns].reset_index(drop=True), frame[TARGET_COLUMN].reset_index(drop=True)


def score_fold(actual: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    """Regression metrics for one fold, all on the ruble scale."""
    return {
        "r2_rub": float(r2_score(actual, predicted)),
        "rmse_rub": float(root_mean_squared_error(actual, predicted)),
        "mae_rub": float(mean_absolute_error(actual, predicted)),
        "median_ae_rub": float(median_absolute_error(actual, predicted)),
        "mape_pct": float(mean_absolute_percentage_error(actual, predicted) * 100),
    }


def cross_validate(x: pd.DataFrame, y: pd.Series, folds: int,
                   seed: int) -> tuple[pd.DataFrame, pd.Series, np.ndarray]:
    """Run k-fold CV and return per-fold metrics, coefficients and OOF predictions."""
    kfold = KFold(n_splits=folds, shuffle=True, random_state=seed)
    rows: list[dict[str, Any]] = []
    predictions = np.full(len(y), np.nan)

    for fold, (train_index, test_index) in enumerate(kfold.split(x), start=1):
        x_train, x_test = x.iloc[train_index], x.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]

        model = LinearRegression()
        model.fit(x_train, y_train)
        predicted = model.predict(x_test)
        predictions[test_index] = predicted

        row: dict[str, Any] = {
            "fold": fold,
            "train_size": len(train_index),
            "test_size": len(test_index),
            "intercept_rub": float(model.intercept_),
        }
        row.update(score_fold(y_test.to_numpy(), predicted))
        for name, value in zip(x.columns, model.coef_):
            row[f"coef_{name}"] = float(value)
        rows.append(row)

    metrics = pd.DataFrame(rows)
    return metrics, y, predictions


def baseline_metrics(y: pd.Series, folds: int, seed: int,
                     x: pd.DataFrame) -> dict[str, float]:
    """Score the mean-predictor baseline with the same folds for comparison."""
    kfold = KFold(n_splits=folds, shuffle=True, random_state=seed)
    rows: list[dict[str, float]] = []
    for train_index, test_index in kfold.split(x):
        baseline = DummyRegressor(strategy="mean")
        baseline.fit(x.iloc[train_index], y.iloc[train_index])
        rows.append(score_fold(
            y.iloc[test_index].to_numpy(), baseline.predict(x.iloc[test_index])
        ))
    return pd.DataFrame(rows).mean(numeric_only=True).to_dict()


def coefficient_table(x: pd.DataFrame, y: pd.Series,
                      metrics: pd.DataFrame) -> pd.DataFrame:
    """Coefficients of the final fit on all data, plus cross-fold stability."""
    model = LinearRegression().fit(x, y)
    rows: list[dict[str, Any]] = []
    for name, value in zip(x.columns, model.coef_):
        fold_values = metrics.get(f"coef_{name}")
        rows.append({
            "term": name,
            "coefficient": float(value),
            "unit": "₽ на единицу",
            "mean_feature": float(x[name].mean()),
            "typical_contribution_rub": float(value * x[name].mean()),
            "fold_coef_std": float(fold_values.std()) if fold_values is not None else np.nan,
        })
    rows.append({
        "term": "intercept",
        "coefficient": float(model.intercept_),
        "unit": "₽",
        "mean_feature": np.nan,
        "typical_contribution_rub": float(model.intercept_),
        "fold_coef_std": float(metrics["intercept_rub"].std()),
    })
    return pd.DataFrame(rows)


def run(args: argparse.Namespace) -> None:
    report_utils.ensure_metrics(METRIC_COLUMNS)
    experiments_dir = args.experiments_dir.expanduser()
    run_id, run_dir = report_utils.create_run_dir(experiments_dir)
    LOG.info("Каталог запуска: %s", run_dir)

    input_path = args.input.expanduser()
    x, y = load_features(input_path)
    if len(x) < args.folds:
        raise SystemExit(f"Строк ({len(x)}) меньше числа фолдов ({args.folds}).")

    metrics, y_all, predictions = cross_validate(x, y, args.folds, args.seed)
    baseline = baseline_metrics(y, args.folds, args.seed, x)
    summary = report_utils.metric_summary(metrics, METRIC_COLUMNS)
    coefficients = coefficient_table(x, y_all, metrics)
    summary_by_metric = summary.set_index("metric")

    predictions_frame = pd.DataFrame({
        "row_index": np.arange(len(y_all)),
        "actual_rub": y_all.to_numpy(dtype=float),
        "predicted_rub": predictions,
    })
    predictions_frame["residual_rub"] = (
        predictions_frame["actual_rub"] - predictions_frame["predicted_rub"]
    )
    predictions_frame["abs_error_rub"] = predictions_frame["residual_rub"].abs()
    diagnostics = report_utils.residual_diagnostics(predictions_frame)
    charts = report_utils.save_figures(
        predictions_frame, run_dir / "figures", title_prefix="Линейная регрессия",
    )

    report_utils.long_metrics(
        metrics, METRIC_COLUMNS, run_id=run_id, model_name=MODEL_NAME,
    ).to_csv(run_dir / "metrics.csv", index=False, encoding="utf-8-sig")
    metrics.to_csv(run_dir / "fold_metrics.csv", index=False, encoding="utf-8-sig")
    summary.to_csv(run_dir / "metrics_summary.csv", index=False, encoding="utf-8-sig")
    coefficients.to_csv(run_dir / "coefficients.csv", index=False, encoding="utf-8-sig")
    predictions_frame.to_csv(run_dir / "predictions.csv", index=False, encoding="utf-8-sig")
    diagnostics.to_csv(run_dir / "residual_diagnostics.csv", index=False, encoding="utf-8-sig")

    best_fold = metrics.loc[metrics["r2_rub"].idxmax()]
    worst_fold = metrics.loc[metrics["r2_rub"].idxmin()]
    intercept = float(coefficients.loc[coefficients["term"].eq("intercept"), "coefficient"].iloc[0])
    low_bias = float(diagnostics.iloc[0]["mean_residual_rub"])
    high_bias = float(diagnostics.iloc[-1]["mean_residual_rub"])

    body = report_utils.build_body(
        [
            f"Модель: `LinearRegression` (МНК, свободный член обучается); "
            f"признаки: **{', '.join(x.columns)}**.",
        ],
        report_utils.metrics_section(summary, METRIC_COLUMNS),
        [
            f"Лучший фолд: №{int(best_fold['fold'])} "
            f"(R² = {report_utils.format_value(best_fold['r2_rub'])}); "
            f"худший: №{int(worst_fold['fold'])} "
            f"(R² = {report_utils.format_value(worst_fold['r2_rub'])}).",
        ],
        report_utils.per_fold_section(
            metrics, ["fold", "train_size", "test_size", *METRIC_COLUMNS],
        ),
        report_utils.baseline_section(
            baseline_rmse=baseline["rmse_rub"],
            model_rmse=summary_by_metric.loc["rmse_rub", "mean"],
            model_label="Регрессия",
            note="Базовая модель предсказывает среднее по обучающей части фолда "
                 "(R² ~ 0 по построению):",
        ),
        [
            "## Коэффициенты (обучение на всех данных)",
            report_utils.markdown_table(coefficients.rename(columns={
                "term": "Термин", "coefficient": "Коэффициент", "unit": "Единица",
                "mean_feature": "Среднее признака",
                "typical_contribution_rub": "Вклад при среднем, ₽",
                "fold_coef_std": "Разброс по фолдам",
            })),
            f"Свободный член отрицательный "
            f"(**{report_utils.format_value(intercept)}** ₽): при малых площадях модель "
            "формально даёт отрицательную цену. Это признак того, что линейная форма плохо "
            "описывает нижний край рынка, а не экономический факт.",
        ],
        report_utils.residual_section(
            diagnostics,
            bias_text=(
                "Смещение (`факт − прогноз`) меняет знак: в нижнем квартиле "
                f"**{report_utils.format_value(low_bias)}** ₽, в верхнем — "
                f"**{report_utils.format_value(high_bias)}** ₽. То есть модель занижает "
                "дешёвые объекты и завышает дорогие: прогноз сжимается к среднему, потому "
                "что два признака не разделяют дорогие сегменты, а квадратичная функция "
                "потерь тянет подгонку к крупным ценам."
            ),
            growth_suffix=": модель заметно хуже на дорогих объектах",
        ),
    )
    limitations = [
        "- `asking_price_rub` — цена предложения, а не подтверждённая цена сделки.",
        "- Только два признака: локация, материалы, транспорт и состояние не учтены,",
        "  поэтому часть дисперсии цены объяснить нечем и она уходит в остаток.",
        "- Цена сильно скошена (skew ~9,8): квадратичная функция потерь делает подгонку",
        "  чувствительной к дорогим объектам, а RMSE — оптимистичной оценкой типичной ошибки.",
        "  Сравнивайте RMSE с MAE и медианной ошибкой выше.",
        "- Линейная модель экстраполирует линейно: для площадей за пределами обучающего",
        "  диапазона прогноз не обоснован.",
        "- Корреляция не означает причинность: коэффициенты описывают связь в этих данных,",
        "  а не прирост цены при надстройке этажа.",
    ]

    artifacts = [
        "report.md", "metrics.csv", "fold_metrics.csv", "metrics_summary.csv",
        "coefficients.csv", "predictions.csv", "residual_diagnostics.csv",
        *[f"figures/{path.name}" for path in charts],
    ]
    manifest = report_utils.build_manifest(
        run_id=run_id,
        script=Path(__file__).name,
        model=MODEL_NAME,
        input_path=input_path,
        rows=len(x),
        features=len(x.columns),
        target=TARGET_COLUMN,
        cv_scheme="kfold",
        folds=args.folds,
        seed=args.seed,
        params=report_utils.cli_params(args),
        artifacts=artifacts,
        primary_value=float(summary_by_metric.loc[report_utils.PRIMARY_METRIC, "mean"]),
    )
    (run_dir / "report.md").write_text(
        report_utils.render_report(
            title="Линейная регрессия цены: 2 признака",
            manifest=manifest, body=body, limitations=limitations,
        ),
        encoding="utf-8",
    )
    index_path = report_utils.finish_run(experiments_dir, run_dir, manifest)

    print(f"Строк: {len(x):,}; признаков: {len(x.columns)}; фолдов: {args.folds}".replace(",", " "))
    # Консоль Windows — cp1251: печатаем только ASCII, иначе UnicodeEncodeError при
    # перенаправлении вывода. «R²» и «±» остаются в отчёте (UTF-8), но не в stdout.
    print(f"R2 (CV): {summary_by_metric.loc['r2_rub', 'mean']:.3f} "
          f"+- {summary_by_metric.loc['r2_rub', 'std']:.3f}")
    print(f"RMSE (CV): {summary_by_metric.loc['rmse_rub', 'mean']:,.0f} руб.".replace(",", " "))
    print(f"MAE (CV): {summary_by_metric.loc['mae_rub', 'mean']:,.0f} руб.".replace(",", " "))
    print(f"Базовая модель RMSE: {baseline['rmse_rub']:,.0f} руб.".replace(",", " "))
    print(f"Эксперимент: {run_dir}; реестр: {index_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Обучение линейной регрессии на признаках из features.csv с кросс-валидацией."
    )
    parser.add_argument(
        "--input", type=Path, default=default_input_path(),
        help="Таблица признаков (по умолчанию data/features/features.csv).",
    )
    parser.add_argument(
        "--experiments-dir", type=Path, default=report_utils.default_experiments_dir(__file__),
        help="Корень экспериментов; запуск получит следующий номер (по умолчанию experiments/).",
    )
    parser.add_argument("--folds", type=int, default=5, help="Число фолдов кросс-валидации (по умолчанию 5).")
    parser.add_argument("--seed", type=int, default=42, help="Случайное зерно перемешивания фолдов.")
    parser.add_argument("--verbose", action="store_true", help="Показывать предупреждения и подробные сообщения.")
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
