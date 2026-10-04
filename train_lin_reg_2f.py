#!/usr/bin/env python3
"""Train a simple linear regression (scikit-learn) and evaluate it with k-fold CV.

Model:    LinearRegression on the numeric feature columns of the input table.
Target:   asking_price_rub (asking price, not a confirmed sale price).
Default:  5-fold KFold cross-validation, reports written under reports/.

The feature set is derived from the input table (every column except the
identifier and the target), so adding features in build_features.py is enough.
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
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    mean_absolute_error,
    mean_absolute_percentage_error,
    median_absolute_error,
    r2_score,
    root_mean_squared_error,
)
from sklearn.model_selection import KFold

import mlflow_utils  # локальный модуль: опциональный трекинг запусков (см. README)


LOG = logging.getLogger("domclick.linreg")
IDENTIFIER_COLUMNS = ("listing_id",)
TARGET_COLUMN = "asking_price_rub"
METRIC_COLUMNS = ("r2", "rmse_rub", "mae_rub", "median_ae_rub", "mape_pct")


def default_input_path() -> Path:
    """Where build_features.py writes the modeling table."""
    return Path(__file__).resolve().parent / "data" / "features" / "features.csv"


def default_report_dir() -> Path:
    """Reports go to reports/, separate from the data."""
    return Path(__file__).resolve().parent / "reports" / "lin_reg_2f"


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
        "r2": float(r2_score(actual, predicted)),
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
    baseline_predictions = np.full(len(y), np.nan)

    for fold, (train_index, test_index) in enumerate(kfold.split(x), start=1):
        x_train, x_test = x.iloc[train_index], x.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]

        model = LinearRegression()
        model.fit(x_train, y_train)
        predicted = model.predict(x_test)
        predictions[test_index] = predicted

        baseline = DummyRegressor(strategy="mean")
        baseline.fit(x_train, y_train)
        baseline_predictions[test_index] = baseline.predict(x_test)

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


def baseline_metrics(y: pd.Series, metrics: pd.DataFrame, folds: int, seed: int,
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


def residual_diagnostics(predictions: pd.DataFrame, quartiles: int = 4) -> pd.DataFrame:
    """Error and bias by prediction quartile, to expose heteroscedasticity."""
    frame = predictions.copy()
    frame["prediction_group"] = pd.qcut(
        frame["predicted_rub"], quartiles,
        labels=[f"Q{index}" for index in range(1, quartiles + 1)],
    )
    grouped = (
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
    return grouped


def save_figures(y: pd.Series, predictions: np.ndarray, figure_dir: Path) -> list[Path]:
    """Out-of-fold diagnostics: predicted vs actual and residuals."""
    figure_dir.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "DejaVu Sans"],
        "axes.unicode_minus": False,
        "figure.dpi": 120,
    })
    actual = y.to_numpy(dtype=float)
    outputs: list[Path] = []

    # Axis limits follow the 99th percentile so outliers do not flatten the cloud.
    # Both values and limits are in millions of rubles — mixing units would clip every point.
    limit = float(np.quantile(np.concatenate([actual, predictions]), 0.99)) / 1e6
    fig, ax = plt.subplots(figsize=(9, 7))
    ax.scatter(actual / 1e6, predictions / 1e6, alpha=0.4, s=18,
               color="#2563eb", edgecolors="none")
    ax.plot([0, limit], [0, limit], color="#dc2626", linewidth=1.4,
            label="идеальное совпадение")
    ax.set(title="Факт и прогноз цены (out-of-fold)", xlabel="Факт, млн ₽",
           ylabel="Прогноз, млн ₽", xlim=(0, limit), ylim=(0, limit))
    ax.legend()
    ax.grid(alpha=0.25)
    fig.tight_layout()
    path = figure_dir / "actual_vs_predicted.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    outputs.append(path)

    residuals = (actual - predictions) / 1e6
    cap = float(np.quantile(np.abs(residuals), 0.99))
    predicted_mln = predictions / 1e6
    x_cap = float(np.quantile(predicted_mln, 0.99))
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.scatter(predicted_mln, residuals, alpha=0.4, s=18,
               color="#7c3aed", edgecolors="none")
    ax.axhline(0, color="#dc2626", linewidth=1.4)
    ax.set(title="Остатки модели (out-of-fold)",
           xlabel="Прогноз, млн ₽", ylabel="Остаток, млн ₽",
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
    return f"{float(value):.3f}".replace(".", ",")


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


def run(input_path: Path, output_dir: Path, folds: int, seed: int,
        args: argparse.Namespace) -> None:
    x, y = load_features(input_path)
    if len(x) < folds:
        raise SystemExit(f"Строк ({len(x)}) меньше числа фолдов ({folds}).")

    metrics, y_all, predictions = cross_validate(x, y, folds, seed)
    baseline = baseline_metrics(y, metrics, folds, seed, x)
    summary = summary_table(metrics)
    coefficients = coefficient_table(x, y_all, metrics)
    summary_by_metric = summary.set_index("metric")

    output_dir.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(output_dir / "cv_metrics.csv", index=False, encoding="utf-8-sig")
    summary.to_csv(output_dir / "cv_summary.csv", index=False, encoding="utf-8-sig")
    coefficients.to_csv(output_dir / "coefficients.csv", index=False, encoding="utf-8-sig")
    predictions_frame = pd.DataFrame({
        "row_index": np.arange(len(y_all)),
        "actual_rub": y_all.to_numpy(dtype=float),
        "predicted_rub": predictions,
    })
    predictions_frame["residual_rub"] = (
        predictions_frame["actual_rub"] - predictions_frame["predicted_rub"]
    )
    predictions_frame["abs_error_rub"] = predictions_frame["residual_rub"].abs()
    predictions_frame.to_csv(output_dir / "predictions.csv", index=False, encoding="utf-8-sig")
    diagnostics = residual_diagnostics(predictions_frame)
    diagnostics.to_csv(output_dir / "residual_diagnostics.csv", index=False, encoding="utf-8-sig")
    charts = save_figures(y_all, predictions, output_dir / "figures")

    def metric_summary(name: str) -> str:
        row = summary_by_metric.loc[name]
        return f"{format_value(row['mean'])} ± {format_value(row['std'])}"

    best_fold = metrics.loc[metrics["r2"].idxmax()]
    worst_fold = metrics.loc[metrics["r2"].idxmin()]
    intercept = float(coefficients.loc[coefficients["term"].eq("intercept"), "coefficient"].iloc[0])
    cheapest = diagnostics.iloc[0]
    dearest = diagnostics.iloc[-1]
    error_gap = dearest["mean_abs_error_rub"] / cheapest["mean_abs_error_rub"]
    report = [
        "# Линейная регрессия цены: 2 признака",
        "",
        f"Данные: `{input_path}`",
        f"Признаки: **{', '.join(x.columns)}**; целевая переменная: **{TARGET_COLUMN}**.",
        f"Модель: `LinearRegression` (МНК, свободный член обучается); строк: **{len(x):,}**.",
        f"Валидация: `KFold(n_splits={folds}, shuffle=True, random_state={seed})`.",
        f"Метрики считаются out-of-fold, в рублях; MAPE — в процентах.",
        "",
        "## Метрики (среднее ± стандартное отклонение по фолдам)",
        "",
        f"- R²: **{metric_summary('r2')}**",
        f"- RMSE: **{metric_summary('rmse_rub')}** ₽",
        f"- MAE: **{metric_summary('mae_rub')}** ₽",
        f"- Медианная абсолютная ошибка: **{metric_summary('median_ae_rub')}** ₽",
        f"- MAPE: **{metric_summary('mape_pct')}** %",
        "",
        f"Лучший фолд: №{int(best_fold['fold'])} (R² = {format_value(best_fold['r2'])}); "
        f"худший: №{int(worst_fold['fold'])} (R² = {format_value(worst_fold['r2'])}).",
        "",
        "## Метрики по фолдам",
        markdown_table(metrics[["fold", "train_size", "test_size", *METRIC_COLUMNS]].rename(
            columns={
                "fold": "Фолд", "train_size": "Обучение", "test_size": "Тест",
                "r2": "R²", "rmse_rub": "RMSE, ₽", "mae_rub": "MAE, ₽",
                "median_ae_rub": "MedAE, ₽", "mape_pct": "MAPE, %",
            }
        )),
        "",
        "## Сравнение с базовой моделью",
        f"Базовая модель предсказывает среднее по обучающей части фолда: "
        f"RMSE **{format_value(baseline['rmse_rub'])}** ₽, MAE "
        f"**{format_value(baseline['mae_rub'])}** ₽, R² ~ 0 по построению.",
        f"Регрессия снижает RMSE относительно базовой модели на "
        f"**{(1 - summary_by_metric.loc['rmse_rub', 'mean'] / baseline['rmse_rub']) * 100:.1f} %**.",
        "",
        "## Коэффициенты (обучение на всех данных)",
        markdown_table(coefficients.rename(columns={
            "term": "Термин", "coefficient": "Коэффициент", "unit": "Единица",
            "mean_feature": "Среднее признака",
            "typical_contribution_rub": "Вклад при среднем, ₽",
            "fold_coef_std": "Разброс по фолдам",
        })),
        f"Свободный член отрицательный (**{format_value(intercept)}** ₽): при малых площадях "
        "модель формально даёт отрицательную цену. Это признак того, что линейная форма "
        "плохо описывает нижний край рынка, а не экономический факт.",
        "",
        "## Диагностика остатков",
        "Средняя абсолютная ошибка и смещение прогноза по квартилям прогноза:",
        markdown_table(diagnostics.rename(columns={
            "prediction_group": "Квартиль прогноза",
            "count": "Строк",
            "mean_predicted_rub": "Средний прогноз, ₽",
            "mean_actual_rub": "Средний факт, ₽",
            "mean_residual_rub": "Среднее смещение, ₽",
            "mean_abs_error_rub": "Средняя |ошибка|, ₽",
        })),
        f"Средняя абсолютная ошибка растёт от нижнего квартиля прогноза "
        f"(**{format_value(cheapest['mean_abs_error_rub'])}** ₽) к верхнему "
        f"(**{format_value(dearest['mean_abs_error_rub'])}** ₽) — в "
        f"**{error_gap:.1f}** раза: модель заметно хуже на дорогих объектах.",
        f"Смещение (`факт − прогноз`) меняет знак: в нижнем квартиле "
        f"**{format_value(cheapest['mean_residual_rub'])}** ₽, в верхнем — "
        f"**{format_value(dearest['mean_residual_rub'])}** ₽. То есть модель занижает "
        "дешёвые объекты и завышает дорогие: прогноз сжимается к среднему, потому что "
        "два признака не разделяют дорогие сегменты, а квадратичная функция потерь "
        "тянет подгонку к крупным ценам.",
        "",
        "## Файлы",
        f"- Метрики по фолдам: `{output_dir / 'cv_metrics.csv'}`",
        f"- Сводка метрик: `{output_dir / 'cv_summary.csv'}`",
        f"- Коэффициенты: `{output_dir / 'coefficients.csv'}`",
        f"- Прогнозы out-of-fold: `{output_dir / 'predictions.csv'}`",
        f"- Диагностика остатков: `{output_dir / 'residual_diagnostics.csv'}`",
        *[f"- График: `{path}`" for path in charts],
        "",
        "## Ограничения",
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
    (output_dir / "report.md").write_text("\n".join(report) + "\n", encoding="utf-8")

    # Модель на всём датасете нужна только MLflow: метрики выше считаются по фолдам.
    final_model = None
    if args.mlflow and args.mlflow_log_model:
        final_model = LinearRegression().fit(x, y)
    mlflow_utils.log_training_run(
        args, model_name="lin-reg-2f", input_path=input_path, output_dir=output_dir,
        extra_params={"rows": len(x), "features": len(x.columns)},
        extra_metrics={"baseline_rmse_rub": float(baseline["rmse_rub"])},
        metrics=metrics, summary=summary, model=final_model,
    )

    print(f"Строк: {len(x):,}; признаков: {len(x.columns)}; фолдов: {folds}".replace(",", " "))
    print(f"R² (CV): {summary_by_metric.loc['r2', 'mean']:.3f} "
          f"± {summary_by_metric.loc['r2', 'std']:.3f}")
    print(f"RMSE (CV): {summary_by_metric.loc['rmse_rub', 'mean']:,.0f} ₽".replace(",", " "))
    print(f"MAE (CV): {summary_by_metric.loc['mae_rub', 'mean']:,.0f} ₽".replace(",", " "))
    print(f"Базовая модель RMSE: {baseline['rmse_rub']:,.0f} ₽".replace(",", " "))
    print(f"Отчёты и графики: {output_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Обучение линейной регрессии на признаках из features.csv с кросс-валидацией."
    )
    parser.add_argument(
        "--input", type=Path, default=default_input_path(),
        help="Таблица признаков (по умолчанию data/features/features.csv).",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=default_report_dir(),
        help="Каталог для отчёта, метрик и figures/ (по умолчанию reports/lin_reg_2f).",
    )
    parser.add_argument("--folds", type=int, default=5, help="Число фолдов кросс-валидации (по умолчанию 5).")
    parser.add_argument("--seed", type=int, default=42, help="Случайное зерно перемешивания фолдов.")
    parser.add_argument("--verbose", action="store_true", help="Показывать предупреждения и подробные сообщения.")
    mlflow_utils.add_mlflow_args(parser)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(levelname)s: %(message)s",
    )
    run(args.input.expanduser(), args.output_dir.expanduser(), args.folds, args.seed, args)


if __name__ == "__main__":
    main()
