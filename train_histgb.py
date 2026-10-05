#!/usr/bin/env python3
"""Train and evaluate a HistGradientBoostingRegressor for the asking-price task.

Model:   sklearn HistGradientBoostingRegressor (gradient boosting on trees) with
         early stopping on an internal validation split.
Data:    data/features/features_rich.csv (produced by build_reach_features.py).
Target:  asking_price_rub, trained on the log scale and scored on the ruble scale.
CV:      GroupKFold by `group_locality` so listings from one locality never split
         between train and test; use --cv kfold only for a quicker comparison.

Why this is not a copy of the linear-regression script
-----------------------------------------------------
1. Missing values are NOT dropped. The wide matrix has NaN in 1659 of 1858 rows
   (distance, settlement counts, badge discount, ...); a row-wise dropna would
   keep only 199 rows and throw away 89 % of the data. HistGradientBoosting
   handles NaN natively, so the matrix is passed through unchanged.
2. The target is modelled on the log scale (price skew ≈ 9.8) and back-transformed
   with expm1 for the ruble-scale metrics, with a clip guard against extrapolation.
3. Grouped CV keeps near-duplicate listings of one locality out of the test set.
4. Results follow the shared experiment contract: a numbered directory under
   `experiments/` with `meta.json`, `report.md` and canonical metric tables; the report
   plumbing lives in `report_utils.py` (see docs/experiment_contract.md).
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Any

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

import report_utils


LOG = logging.getLogger("domclick.histgb")
MODEL_NAME = "histgb"
IDENTIFIER_COLUMN = "listing_id"
TARGET_COLUMN = "asking_price_rub"
LOG_TARGET_COLUMN = "log_asking_price_rub"
GROUP_COLUMN = "group_locality"

# id, both target variants and the CV group key must never enter the design matrix.
NON_FEATURE_COLUMNS = (IDENTIFIER_COLUMN, TARGET_COLUMN, LOG_TARGET_COLUMN, GROUP_COLUMN)

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


def load_data(path: Path) -> tuple[pd.DataFrame, pd.Series, pd.Series, pd.Series, list[str]]:
    """Read the matrix and split it into features, target, CV groups and identifiers.

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

        params: dict[str, Any] = {}
        if args.tune:
            params, tuning = tune_params(
                x_train, y_train_log, groups.iloc[train_index],
                args.tune, args.inner_folds, args.seed + fold, args.max_iter,
            )
            tuning.insert(0, "fold", fold)
            tuning_frames.append(tuning)

        model = make_model(params, args.seed, args.max_iter)
        model.fit(x_train, y_train_log)
        predicted_log = model.predict(x_test)
        predicted_rub = np.expm1(predicted_log)

        # Trees cannot extrapolate far, but the log back-transform still can; clip to the
        # range seen in training so a single wild prediction cannot dominate the metrics.
        clipped = 0
        if args.clip_predictions:
            low, high = y.iloc[train_index].min(), y.iloc[train_index].max()
            clipped_values = np.clip(predicted_rub, low, high)
            clipped = int(np.sum(clipped_values != predicted_rub))
            predicted_rub = clipped_values
        predictions[test_index] = predicted_rub

        baseline = DummyRegressor(strategy="mean")
        baseline.fit(x_train, y_train_log)
        baseline_predictions[test_index] = np.expm1(baseline.predict(x_test))

        row: dict[str, Any] = {
            "fold": fold,
            "train_size": len(train_index),
            "test_size": len(test_index),
            "n_iter": int(model.n_iter_),
            "clipped": clipped,
        }
        row.update(score_fold(y.iloc[test_index].to_numpy(dtype=float), predicted_rub,
                              y_log[test_index], predicted_log))
        rows.append(row)
        print(f"Фолд {fold}/{args.folds}: R2(руб.) = {row['r2_rub']:.3f}, "
              f"MAE = {row['mae_rub']:,.0f}, итераций = {row['n_iter']}".replace(",", " "))

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


def run(args: argparse.Namespace) -> None:
    report_utils.ensure_metrics(METRIC_COLUMNS)
    experiments_dir = args.experiments_dir.expanduser()
    run_id, run_dir = report_utils.create_run_dir(experiments_dir)
    LOG.info("Каталог запуска: %s", run_dir)
    input_path: Path = args.input
    x, y, groups, identifiers, feature_columns = load_data(input_path)

    if args.cv == "group" and groups.nunique() < args.folds:
        raise SystemExit(
            f"Групп ({groups.nunique()}) меньше числа фолдов ({args.folds}); "
            "используйте --cv kfold или уменьшите --folds."
        )
    if len(x) < args.folds:
        raise SystemExit(f"Строк ({len(x)}) меньше числа фолдов ({args.folds}).")

    metrics, predictions, baseline, importance, tuning = cross_validate(x, y, groups, args)
    summary = report_utils.metric_summary(metrics, METRIC_COLUMNS)
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

    diagnostics = report_utils.residual_diagnostics(oof)
    charts = report_utils.save_figures(oof, run_dir / "figures", title_prefix="HistGB")
    report_utils.long_metrics(
        metrics, METRIC_COLUMNS, run_id=run_id, model_name=MODEL_NAME,
    ).to_csv(run_dir / "metrics.csv", index=False, encoding="utf-8-sig")
    metrics.to_csv(run_dir / "fold_metrics.csv", index=False, encoding="utf-8-sig")
    summary.to_csv(run_dir / "metrics_summary.csv", index=False, encoding="utf-8-sig")
    oof.to_csv(run_dir / "predictions.csv", index=False, encoding="utf-8-sig")
    diagnostics.to_csv(run_dir / "residual_diagnostics.csv", index=False, encoding="utf-8-sig")
    if not importance.empty:
        importance.to_csv(run_dir / "feature_importance.csv", index=False, encoding="utf-8-sig")
    if not tuning.empty:
        tuning.to_csv(run_dir / "tuning.csv", index=False, encoding="utf-8-sig")

    best_fold = metrics.loc[metrics["r2_rub"].idxmax()]
    worst_fold = metrics.loc[metrics["r2_rub"].idxmin()]
    baseline_rmse = root_mean_squared_error(oof["actual_rub"], oof["baseline_rub"])
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

    body = report_utils.build_body(
        [
            f"Модель: `HistGradientBoostingRegressor` с early stopping "
            f"(`max_iter={args.max_iter}`, `n_iter_no_change=20`, `validation_fraction=0.1`); "
            f"признаков: **{len(feature_columns)}**; пропусков в матрице: **{nan_cells:,}**.",
            f"Цель: `log({TARGET_COLUMN})`; метрики пересчитаны в рубли через `expm1`; "
            f"групп в валидации: {groups.nunique():,}.",
            f"Подбор гиперпараметров: "
            f"{'внутренний перебор на ' + str(args.tune) + ' кандидатах' if args.tune else 'не выполнялся'}.",
        ],
        report_utils.metrics_section(summary, METRIC_COLUMNS),
        [
            f"Лучший фолд: №{int(best_fold['fold'])} "
            f"(R² в рублях = {report_utils.format_value(best_fold['r2_rub'])}); "
            f"худший: №{int(worst_fold['fold'])} "
            f"(R² в рублях = {report_utils.format_value(worst_fold['r2_rub'])}).",
        ],
        report_utils.per_fold_section(
            metrics, ["fold", "train_size", "test_size", "n_iter", "clipped", *METRIC_COLUMNS],
        ),
        report_utils.baseline_section(
            baseline_rmse=baseline_rmse,
            model_rmse=summary_by_metric.loc["rmse_rub", "mean"],
            model_label="HistGB",
            note="Базовая модель предсказывает среднее по обучающей части фолда "
                 "(обратное преобразование лога — геометрическое среднее):",
        ),
        report_utils.importance_section(importance),
        report_utils.tuning_section(
            tuning, ["fold", "rmse_log_mean", "learning_rate", "max_leaf_nodes",
                     "min_samples_leaf", "l2_regularization"],
        ),
        report_utils.residual_section(
            diagnostics,
            conclusion="Остаточная гетероскедастичность ожидаема: дорогие объекты "
                       "разнороднее, а обучающих примеров в верхнем сегменте мало.",
        ),
        report_utils.text_section("Обработка пропусков", [
            "Пропуски **не заполняются**: `HistGradientBoostingRegressor` сам направляет "
            "объекты с NaN в отдельную ветвь. Это принципиально, потому что в матрице "
            f"**{nan_cells:,}** пропущенных ячеек, и хотя бы один пропуск есть в "
            f"**{rows_with_nan:,}** из **{len(x):,}** строк "
            f"({rows_with_nan / len(x) * 100:.1f} %). Построчное удаление пропусков "
            f"оставило бы **{len(x) - rows_with_nan:,}** строк — HistGB сохраняет все данные.",
        ]),
    )
    limitations = [
        "- `asking_price_rub` — цена предложения, а не подтверждённая цена сделки.",
        cv_note,
        "- Целевая переменная скошена (skew ≈ 9,8): RMSE сильно зависит от нескольких "
        "дорогих объектов, поэтому рядом приведены MAE и медианная ошибка.",
        "- Признак `badge_mortgage_discount_pct` — прокси продавца/сегмента "
        "(корреляция с ценой 0,82, немонотонен по цене). Проверяйте устойчивость "
        "результата без группы `badge_*`.",
        "- Прогнозы обрезаются по диапазону обучающей части фолда "
        f"({'включено' if args.clip_predictions else 'выключено'}), "
        "поэтому модель не выдаёт цены за пределами наблюдённого диапазона.",
        "- Модель не экстраполирует: для объектов вне обучающего распределения "
        "(другой регион, другой тип недвижимости) прогноз не обоснован.",
    ]
    artifacts = [
        "report.md", "metrics.csv", "fold_metrics.csv", "metrics_summary.csv",
        "predictions.csv", "residual_diagnostics.csv",
        *(["feature_importance.csv"] if not importance.empty else []),
        *(["tuning.csv"] if not tuning.empty else []),
        *[f"figures/{path.name}" for path in charts],
    ]
    manifest = report_utils.build_manifest(
        run_id=run_id, script=Path(__file__).name, model=MODEL_NAME, input_path=input_path,
        rows=len(x), features=len(feature_columns), target=TARGET_COLUMN,
        group_column=GROUP_COLUMN, cv_scheme=args.cv, folds=args.folds, seed=args.seed,
        params=report_utils.cli_params(args), artifacts=artifacts,
        primary_value=float(summary_by_metric.loc[report_utils.PRIMARY_METRIC, "mean"]),
    )
    (run_dir / "report.md").write_text(
        report_utils.render_report(
            title="HistGradientBoosting: модель цены предложения",
            manifest=manifest, body=body, limitations=limitations,
        ),
        encoding="utf-8",
    )
    index_path = report_utils.finish_run(experiments_dir, run_dir, manifest)

    print(f"Rows: {len(x):,}; features: {len(feature_columns)}; folds: {args.folds}".replace(",", " "))
    print(f"R2 (rub): {summary_by_metric.loc['r2_rub', 'mean']:.3f} "
          f"+- {summary_by_metric.loc['r2_rub', 'std']:.3f}")
    print(f"R2 (log): {summary_by_metric.loc['r2_log', 'mean']:.3f}")
    print(f"RMSE (rub): {summary_by_metric.loc['rmse_rub', 'mean']:,.0f}".replace(",", " "))
    print(f"MAE (rub): {summary_by_metric.loc['mae_rub', 'mean']:,.0f}".replace(",", " "))
    print(f"Baseline RMSE (rub): {baseline_rmse:,.0f}".replace(",", " "))
    print(f"Эксперимент: {run_dir}; реестр: {index_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Обучение HistGradientBoosting на features_rich.csv с кросс-валидацией."
    )
    parser.add_argument(
        "--input", type=Path, default=default_input_path(),
        help="Матрица признаков (по умолчанию data/features/features_rich.csv).",
    )
    parser.add_argument(
        "--experiments-dir", type=Path, default=report_utils.default_experiments_dir(__file__),
        help="Корень экспериментов; запуск получит следующий номер (по умолчанию experiments/).",
    )
    parser.add_argument(
        "--cv", choices=("group", "kfold"), default="group",
        help="'group' — GroupKFold по населённому пункту (честно), 'kfold' — быстрее, но оптимистично.",
    )
    parser.add_argument("--folds", type=int, default=5, help="Число фолдов (по умолчанию 5).")
    parser.add_argument("--seed", type=int, default=42, help="Случайное зерно.")
    parser.add_argument("--max-iter", type=int, default=400, help="Максимум итераций бустинга.")
    parser.add_argument("--tune", type=int, default=0,
                        help="Число случайных кандидатов гиперпараметров на фолд (0 — без подбора).")
    parser.add_argument("--inner-folds", type=int, default=3,
                        help="Фолдов во внутренней разбивке при подборе гиперпараметров.")
    parser.add_argument("--importance-repeats", type=int, default=5,
                        help="Повторов permutation importance (0 — не считать).")
    parser.add_argument("--no-clip-predictions", dest="clip_predictions", action="store_false",
                        default=True,
                        help="Отключить обрезку прогнозов по диапазону обучающей части фолда.")
    parser.add_argument("--verbose", action="store_true",
                        help="Показывать предупреждения и подробные сообщения.")
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
