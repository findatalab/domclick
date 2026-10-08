#!/usr/bin/env python3
"""Рисунки для статьи о промптах извлечения признаков.

Скрипт не обучает моделей и не читает сырые выгрузки: он собирает готовые к вставке в
статью изображения из уже сохранённых артефактов — реестра запусков (`experiments/`),
таблиц EDA (`reports/eda/`) и матриц признаков (`data/features/`). Результат по
умолчанию складывается в `docs/figures/`, откуда рисунки подключаются в
`docs/paper_plan.md`. Повторный запуск детерминированно перезаписывает файлы.

Конфигурации статьи сопоставлены с запусками реестра через `CONFIGURATIONS`;
`validate_configurations` сверяет номер запуска с ожидаемым числом признаков, поэтому
случайное переименование каталогов не приведёт к молчаливой подмене данных.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.figure import Figure
from matplotlib.ticker import FuncFormatter

REPO_ROOT = Path(__file__).resolve().parent
EXPERIMENTS_DIR = REPO_ROOT / "experiments"
FEATURES_DIR = REPO_ROOT / "data" / "features"
PRICE_DISTRIBUTION_CSV = REPO_ROOT / "reports" / "eda" / "price_distribution.csv"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "docs" / "figures"

MILLION = 1_000_000
WORKING_PRICE_LIMIT_RUB = 50_000_000
DEFAULT_DPI = 200


@dataclass(frozen=True)
class Configuration:
    """Конфигурация представления статьи и соответствующий ей запуск реестра."""

    name: str
    label: str
    run_id: str
    features_file: str


CONFIGURATIONS: tuple[Configuration, ...] = (
    Configuration("features_v0", "v0", "01", "features_v0.csv"),
    Configuration("features_v1", "v1", "02", "features_v1.csv"),
    Configuration("features_v2", "v2", "03", "features_v2.csv"),
)

# Число признаков из meta.json запуска; ловит переименование каталогов experiments/.
EXPECTED_FEATURES = {"v0": 81, "v1": 80, "v2": 62}

COLORS = {"v0": "#64748b", "v1": "#2563eb", "v2": "#0f766e"}
ACCENT = "#dc2626"

# Русские подписи признаков для рисунка важности; неизвестные остаются как в отчёте.
FEATURE_LABELS = {
    "house_area_m2": "площадь дома",
    "distance_to_mkad_km": "расстояние от МКАД",
    "house_area_per_floor": "площадь на этаж",
    "bathrooms": "число санузлов",
    "land_area_sotka": "площадь участка",
    "land_to_house_ratio": "участок к дому",
    "finishing_stage_without_repair": "отделка отсутствует",
    "finishing_stage_with_repair": "отделка есть",
    "highway_other": "шоссе: прочие",
    "highway_00": "шоссе: Боровское",
    "highway_01": "шоссе: Варшавское",
    "highway_02": "шоссе: Волоколамское",
    "highway_03": "шоссе: Калужское",
    "highway_04": "шоссе: Киевское",
    "highway_05": "шоссе: Минское",
    "highway_06": "шоссе: Симферопольское",
    "sewage_type_otsutstvuet": "канализация отсутствует",
    "sewage_type_centralnoe": "канализация центральная",
    "sewage_type_septik": "канализация: септик",
    "land_category_izhs": "категория: ИЖС",
    "land_category_snt": "категория: СНТ",
    "land_category_lph": "категория: ЛПХ",
    "gas_status_podklyuchen": "газ подключён",
    "gas_status_otsutstvuet": "газ отсутствует",
    "gas_status_po_granice": "газ по границе",
    "heating_type_pechnoe": "отопление печное",
    "heating_type_gazovoe": "отопление газовое",
    "heating_type_elektricheskoe": "отопление электрическое",
    "wall_material_karkas": "стены: каркас",
    "wall_material_kirpich": "стены: кирпич",
    "wall_material_unknown": "стены не указаны",
    "water_source_centralnoe": "водоснабжение центральное",
    "water_source_skvazhina": "водоснабжение: скважина",
}

# (подпись, колонка features_v1, вид) — покрытие характеристик, извлечённых из текста.
COVERAGE_FIELDS = (
    ("Отопление", "heating_type_unknown", "level"),
    ("Отделка", "finishing_stage_unknown", "level"),
    ("Канализация", "sewage_type_unknown", "level"),
    ("Водоснабжение", "water_source_unknown", "level"),
    ("Газ", "gas_status_unknown", "level"),
    ("Фундамент", "foundation_type_unknown", "level"),
    ("Стены", "wall_material_unknown", "level"),
    ("Категория земли", "land_category_unknown", "level"),
    ("Год постройки", "construction_year", "numeric"),
    ("Площадь бани", "bathhouse_area_m2", "numeric"),
    ("Площадь гаража", "garage_area_m2", "numeric"),
    ("Площадь гостевого дома", "guest_house_area_m2", "numeric"),
)


def comma(value: float, digits: int = 2) -> str:
    """Число с запятой как десятичным разделителем (русская типографика)."""
    return f"{value:.{digits}f}".replace(".", ",")


def comma_axis(digits: int = 0) -> FuncFormatter:
    """Форматтер делений оси с запятой вместо точки."""
    return FuncFormatter(lambda value, _position: comma(value, digits))


def configure_style() -> None:
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "DejaVu Sans"],
        "axes.unicode_minus": False,
        "figure.dpi": 120,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "axes.titlesize": 12,
        "axes.labelsize": 11,
        "figure.titlesize": 14,
    })


def run_dir(run_id: str) -> Path:
    path = EXPERIMENTS_DIR / run_id
    if not path.is_dir():
        raise SystemExit(f"Не найден каталог запуска: {path}")
    return path


def read_meta(run_id: str) -> dict:
    return json.loads((run_dir(run_id) / "meta.json").read_text(encoding="utf-8"))


def read_fold_metrics(run_id: str) -> pd.DataFrame:
    return pd.read_csv(run_dir(run_id) / "fold_metrics.csv")


def read_predictions(run_id: str) -> pd.DataFrame:
    return pd.read_csv(run_dir(run_id) / "predictions.csv")


def read_residual_diagnostics(run_id: str) -> pd.DataFrame:
    return pd.read_csv(run_dir(run_id) / "residual_diagnostics.csv")


def read_metrics_summary(run_id: str) -> dict[str, float]:
    """Средние по фолдам метрики запуска — те же числа, что в таблице статьи."""
    summary = pd.read_csv(run_dir(run_id) / "metrics_summary.csv")
    return {row.metric: float(row.mean) for row in summary.itertuples()}


def read_importance(run_id: str) -> pd.DataFrame:
    return pd.read_csv(run_dir(run_id) / "feature_importance.csv")


def read_working_prices() -> pd.Series:
    prices = pd.read_csv(FEATURES_DIR / CONFIGURATIONS[0].features_file,
                         usecols=["asking_price_rub"])["asking_price_rub"]
    return prices.loc[prices.le(WORKING_PRICE_LIMIT_RUB)] / MILLION


def validate_configurations() -> None:
    for config in CONFIGURATIONS:
        expected = EXPECTED_FEATURES[config.label]
        actual = int(read_meta(config.run_id)["data"]["features"])
        if actual != expected:
            raise SystemExit(
                f"Запуск {config.run_id} не соответствует {config.name}: ожидалось "
                f"{expected} признаков, в meta.json — {actual}. Обновите CONFIGURATIONS."
            )


def save(fig: Figure, output_dir: Path, name: str, dpi: int) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / name
    fig.savefig(path, dpi=dpi, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    return path


def figure_price_distribution(output_dir: Path, dpi: int) -> Path:
    binned = pd.read_csv(PRICE_DISTRIBUTION_CSV)
    working = read_working_prices()

    fig, (ax_full, ax_work) = plt.subplots(1, 2, figsize=(12, 4.8))

    positions = np.arange(len(binned))
    ax_full.bar(positions, binned["listing_count"], color=COLORS["v1"], edgecolor="white")
    ax_full.set_xticks(positions, binned["price_range"], rotation=45, ha="right")
    for position, count in zip(positions, binned["listing_count"]):
        ax_full.annotate(int(count), (position, count), ha="center", va="bottom", fontsize=8)
    ax_full.set_title(f"(а) Полный массив, n = {int(binned['listing_count'].sum())}")
    ax_full.set_xlabel("Цена предложения")
    ax_full.set_ylabel("Количество объявлений")
    ax_full.set_ylim(0, float(binned["listing_count"].max()) * 1.15)

    ax_work.hist(working, bins=30, color=COLORS["v2"], edgecolor="white")
    median = float(working.median())
    ax_work.axvline(median, color=ACCENT, linewidth=1.5,
                    label=f"медиана {comma(median)} млн ₽")
    ax_work.set_title(f"(б) Рабочий сегмент, n = {len(working)}")
    ax_work.set_xlabel("Цена предложения, млн ₽")
    ax_work.set_ylabel("Количество объявлений")
    ax_work.legend()

    fig.suptitle("Распределение цен предложения")
    fig.tight_layout()
    return save(fig, output_dir, "fig01_price_distribution.png", dpi)


def figure_model_comparison(output_dir: Path, dpi: int) -> Path:
    labels: list[str] = []
    rmse_mean: list[float] = []
    rmse_std: list[float] = []
    mape_mean: list[float] = []
    mape_std: list[float] = []
    for config in CONFIGURATIONS:
        folds = read_fold_metrics(config.run_id)
        labels.append(config.label)
        rmse_mean.append(float(folds["rmse_rub"].mean()) / MILLION)
        rmse_std.append(float(folds["rmse_rub"].std(ddof=1)) / MILLION)
        mape_mean.append(float(folds["mape_pct"].mean()))
        mape_std.append(float(folds["mape_pct"].std(ddof=1)))

    positions = np.arange(len(CONFIGURATIONS))
    colors = [COLORS[label] for label in labels]
    fig, (ax_rmse, ax_mape) = plt.subplots(1, 2, figsize=(12, 4.8))

    ax_rmse.bar(positions, rmse_mean, yerr=rmse_std, capsize=5, color=colors,
                edgecolor="white")
    for position, value, deviation in zip(positions, rmse_mean, rmse_std):
        ax_rmse.annotate(comma(value, 3), (position, value + deviation),
                         ha="center", va="bottom", fontsize=9)
    for position, value in zip(positions[1:], rmse_mean[1:]):
        drop = (rmse_mean[0] - value) / rmse_mean[0] * 100
        ax_rmse.annotate(f"−{comma(drop)} %", (position, value * 0.5), ha="center",
                         va="center", color="white", fontsize=9, fontweight="bold")
    ax_rmse.set_xticks(positions, labels)
    ax_rmse.set_title("(а) RMSE, среднее ± стандартное отклонение по фолдам")
    ax_rmse.set_xlabel("Конфигурация представления")
    ax_rmse.set_ylabel("RMSE, млн ₽")
    ax_rmse.set_ylim(0, max(m + s for m, s in zip(rmse_mean, rmse_std)) * 1.15)
    ax_rmse.yaxis.set_major_formatter(comma_axis(0))

    ax_mape.bar(positions, mape_mean, yerr=mape_std, capsize=5, color=colors,
                edgecolor="white")
    for position, value, deviation in zip(positions, mape_mean, mape_std):
        ax_mape.annotate(comma(value, 1), (position, value + deviation),
                         ha="center", va="bottom", fontsize=9)
    ax_mape.set_xticks(positions, labels)
    ax_mape.set_title("(б) MAPE, среднее ± стандартное отклонение по фолдам")
    ax_mape.set_xlabel("Конфигурация представления")
    ax_mape.set_ylabel("MAPE, %")
    ax_mape.set_ylim(0, max(m + s for m, s in zip(mape_mean, mape_std)) * 1.15)
    ax_mape.yaxis.set_major_formatter(comma_axis(0))

    fig.suptitle("Сравнение конфигураций представления")
    fig.tight_layout()
    return save(fig, output_dir, "fig02_model_comparison.png", dpi)


def figure_actual_vs_predicted(output_dir: Path, dpi: int) -> Path:
    panels = (CONFIGURATIONS[0], CONFIGURATIONS[-1])
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.4))
    for index, (ax, config) in enumerate(zip(axes, panels)):
        predictions = read_predictions(config.run_id)
        actual = predictions["actual_rub"].to_numpy(dtype=float) / MILLION
        predicted = predictions["predicted_rub"].to_numpy(dtype=float) / MILLION
        limit = float(np.quantile(np.concatenate([actual, predicted]), 0.99))

        ax.scatter(actual, predicted, alpha=0.4, s=16, color=COLORS[config.label],
                   edgecolors="none")
        ax.plot([0, limit], [0, limit], color=ACCENT, linewidth=1.4,
                label="идеальное совпадение")

        summary = read_metrics_summary(config.run_id)
        rmse = summary["rmse_rub"] / MILLION
        ax.annotate(f"R² = {comma(summary['r2_rub'], 3)}\nRMSE = {comma(rmse, 3)} млн ₽",
                    xy=(0.04, 0.95),
                    xycoords="axes fraction", va="top", fontsize=10,
                    bbox=dict(boxstyle="round", facecolor="white", alpha=0.85,
                              edgecolor="#cbd5e1"))

        ax.set_title(f"({chr(1072 + index)}) {config.name}, {len(predictions)} наблюдений")
        ax.set_xlabel("Факт, млн ₽")
        ax.set_ylabel("Прогноз, млн ₽")
        ax.set_xlim(0, limit)
        ax.set_ylim(0, limit)
        ax.legend(loc="lower right")

    fig.suptitle("Факт и прогноз цены, вневыборочные прогнозы (до 99-го перцентиля)")
    fig.tight_layout()
    return save(fig, output_dir, "fig03_actual_vs_predicted.png", dpi)


def figure_accuracy_vs_size(output_dir: Path, dpi: int) -> Path:
    points: list[tuple[int, float, Configuration]] = []
    for config in CONFIGURATIONS:
        folds = read_fold_metrics(config.run_id)
        points.append((int(read_meta(config.run_id)["data"]["features"]),
                       float(folds["rmse_rub"].mean()) / MILLION, config))

    fig, ax = plt.subplots(figsize=(7.6, 5.2))
    xs = [point[0] for point in points]
    ys = [point[1] for point in points]
    for x, y, config in points:
        ax.scatter(x, y, s=110, color=COLORS[config.label], zorder=3, edgecolors="white")
        # v0 и v1 отличаются одним признаком: подписи разводим в стороны, чтобы не слиплись.
        if x == max(xs):
            offset, align = (10, 2), "left"
        elif x == min(xs):
            offset, align = (0, 14), "center"
        else:
            offset, align = (-10, 2), "right"
        ax.annotate(f"{config.name}\n{comma(y, 3)} млн ₽", (x, y),
                    textcoords="offset points", xytext=offset, ha=align, fontsize=9)
    for (x0, y0, _), (x1, y1, _) in zip(points, points[1:]):
        ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                    arrowprops=dict(arrowstyle="->", color="#94a3b8", linewidth=1.4))

    ax.set_xlabel("Число признаков")
    ax.set_ylabel("RMSE, млн ₽ (среднее по фолдам)")
    ax.set_title("Точность и размер представления")
    span = max(xs) - min(xs)
    ax.set_xticks(sorted(xs))
    ax.set_xlim(min(xs) - span * 0.2, max(xs) + span * 0.2)
    ax.set_ylim(min(ys) - 0.2, max(ys) + 0.35)
    ax.yaxis.set_major_formatter(comma_axis(1))
    fig.tight_layout()
    return save(fig, output_dir, "fig04_accuracy_vs_size.png", dpi)


def figure_feature_importance(output_dir: Path, dpi: int, top_n: int = 15) -> Path:
    config = CONFIGURATIONS[-1]
    column = "importance_mean_r2_log"
    importance = (read_importance(config.run_id)
                  .sort_values(column, ascending=False)
                  .head(top_n)
                  .iloc[::-1])

    values = importance[column].to_numpy(dtype=float)
    labels = [FEATURE_LABELS.get(feature, feature) for feature in importance["feature"]]
    colors = ["#0f766e" if rank >= len(values) - 3 else "#94a3b8" for rank in range(len(values))]

    fig, ax = plt.subplots(figsize=(9.2, 6.4))
    ax.barh(np.arange(len(values)), values, color=colors, height=0.72)
    ax.set_yticks(np.arange(len(values)), labels)
    for position, value in enumerate(values):
        ax.annotate(comma(value, 3), (value, position), xytext=(4, 0),
                    textcoords="offset points", va="center", fontsize=8)
    ax.set_xlabel("Снижение R² на логарифмической шкале")
    ax.set_title(f"Важность признаков {config.name} (топ-{top_n})")
    ax.set_xlim(0, float(values.max()) * 1.18)
    ax.xaxis.set_major_formatter(comma_axis(2))
    fig.tight_layout()
    return save(fig, output_dir, "fig05_feature_importance.png", dpi)


def figure_residuals(output_dir: Path, dpi: int) -> Path:
    config = CONFIGURATIONS[-1]
    predictions = read_predictions(config.run_id)
    actual = predictions["actual_rub"].to_numpy(dtype=float)
    predicted_rub = predictions["predicted_rub"].to_numpy(dtype=float)
    predicted = predicted_rub / MILLION
    residual = (actual - predicted_rub) / MILLION
    diagnostics = read_residual_diagnostics(config.run_id)

    fig, (ax_scatter, ax_group) = plt.subplots(1, 2, figsize=(12, 5.0))

    cap = float(np.quantile(np.abs(residual), 0.99))
    ax_scatter.scatter(predicted, residual, alpha=0.4, s=16, color=COLORS[config.label],
                       edgecolors="none")
    ax_scatter.axhline(0, color=ACCENT, linewidth=1.4)
    ax_scatter.set_title("(а) Остатки по прогнозу")
    ax_scatter.set_xlabel("Прогноз, млн ₽")
    ax_scatter.set_ylabel("Остаток (факт − прогноз), млн ₽")
    ax_scatter.set_xlim(0, float(np.quantile(predicted, 0.99)))
    ax_scatter.set_ylim(-cap, cap)
    ax_scatter.yaxis.set_major_formatter(comma_axis(0))

    positions = np.arange(len(diagnostics))
    mae = (diagnostics["mean_abs_error_rub"] / MILLION).to_numpy(dtype=float)
    bias = (diagnostics["mean_residual_rub"] / MILLION).to_numpy(dtype=float)
    width = 0.38
    ax_group.bar(positions - width / 2, mae, width=width, color=COLORS[config.label],
                 label="средняя абсолютная ошибка")
    ax_group.bar(positions + width / 2, bias, width=width, color="#f59e0b",
                 label="смещение (факт − прогноз)")
    for position, value in zip(positions - width / 2, mae):
        ax_group.annotate(comma(value), (position, value), ha="center", va="bottom", fontsize=8)
    for position, value in zip(positions + width / 2, bias):
        ax_group.annotate(comma(value), (position, value), ha="center", va="bottom", fontsize=8)
    ax_group.set_xticks(positions, [f"{group}\n({count})" for group, count
                                    in zip(diagnostics["prediction_group"], diagnostics["count"])])
    ax_group.set_title("(б) Ошибка и смещение по квартилям прогноза")
    ax_group.set_xlabel("Квартиль прогноза (число наблюдений)")
    ax_group.set_ylabel("млн ₽")
    ax_group.set_ylim(0, max(mae) * 1.2)
    ax_group.legend(fontsize=9)

    fig.suptitle(f"Диагностика ошибок {config.name}, вневыборочные прогнозы")
    fig.tight_layout()
    return save(fig, output_dir, "fig06_residuals.png", dpi)


def figure_llm_coverage(output_dir: Path, dpi: int) -> Path:
    config = CONFIGURATIONS[1]
    features = pd.read_csv(FEATURES_DIR / config.features_file)

    rows: list[tuple[str, float]] = []
    for label, column, kind in COVERAGE_FIELDS:
        if column not in features.columns:
            continue
        series = features[column]
        share = (series.notna().mean() if kind == "numeric"
                 else 1.0 - series.fillna(0).astype(bool).mean())
        rows.append((label, float(share) * 100))
    frame = pd.DataFrame(rows, columns=["label", "share"]).sort_values("share")

    fig, ax = plt.subplots(figsize=(9.0, 6.0))
    ax.barh(frame["label"], frame["share"], color="#0891b2", height=0.72)
    for position, value in enumerate(frame["share"]):
        ax.annotate(f"{comma(value, 1)} %", (value, position), xytext=(4, 0),
                    textcoords="offset points", va="center", fontsize=8)
    ax.set_xlim(0, 100)
    ax.set_xlabel("Доля объявлений с установленным значением, %")
    ax.set_title(f"Покрытие извлечённых характеристик ({config.name})")
    fig.tight_layout()
    return save(fig, output_dir, "fig07_llm_coverage.png", dpi)


BUILDERS: tuple[Callable[[Path, int], Path], ...] = (
    figure_price_distribution,
    figure_model_comparison,
    figure_actual_vs_predicted,
    figure_accuracy_vs_size,
    figure_feature_importance,
    figure_residuals,
    figure_llm_coverage,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR,
                        help="каталог для изображений (по умолчанию docs/figures)")
    parser.add_argument("--dpi", type=int, default=DEFAULT_DPI,
                        help="разрешение сохранения, точек на дюйм")
    return parser.parse_args()


def display(path: Path) -> str:
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def main() -> None:
    args = parse_args()
    if args.dpi <= 0:
        raise SystemExit("--dpi должен быть положительным")
    configure_style()
    validate_configurations()
    output_dir = args.output_dir if args.output_dir.is_absolute() else REPO_ROOT / args.output_dir
    for builder in BUILDERS:
        path = builder(output_dir, args.dpi)
        print(f"Сохранено: {display(path)}")


if __name__ == "__main__":
    main()
