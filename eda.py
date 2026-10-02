#!/usr/bin/env python3
"""Exploratory analysis of the Domclick objects.jsonl dataset."""

from __future__ import annotations

import argparse
import json
import logging
import math
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


LOG = logging.getLogger("domclick.eda")
NUMERIC_COLUMNS = (
	"asking_price_rub",
	"price_per_m2_advertised",
	"price_per_m2_computed",
	"house_area_m2",
	"land_area_sotka",
	"floors",
	"distance_to_mkad_km",
	"photo_count_advertised",
)
PRICE_BINS = [0, 5_000_000, 10_000_000, 15_000_000, 20_000_000,
			  30_000_000, 50_000_000, 75_000_000, 100_000_000, np.inf]
PRICE_BIN_LABELS = ["< 5 млн", "5–10 млн", "10–15 млн", "15–20 млн",
					"20–30 млн", "30–50 млн", "50–75 млн", "75–100 млн",
					"≥ 100 млн"]


def default_input_path() -> Path:
	"""Find the generated dataset in the usual project output locations."""
	root = Path(__file__).resolve().parent
	candidates = (root / "data" / "dataset" / "objects.jsonl", )
	for candidate in candidates:
		if candidate.is_file():
			return candidate
	return candidates[0]


def read_jsonl(path: Path) -> tuple[pd.DataFrame, int]:
	"""Read JSONL records and skip malformed/non-object lines with a warning."""
	rows: list[dict[str, Any]] = []
	invalid_lines = 0
	try:
		with path.open("r", encoding="utf-8-sig") as stream:
			for line_number, line in enumerate(stream, start=1):
				if not line.strip():
					continue
				try:
					record = json.loads(line)
				except json.JSONDecodeError as exc:
					invalid_lines += 1
					LOG.warning("Некорректный JSON в строке %s: %s", line_number, exc)
					continue
				if not isinstance(record, dict):
					invalid_lines += 1
					LOG.warning("Строка %s содержит JSON не в виде объекта; пропущена", line_number)
					continue
				rows.append(record)
	except OSError as exc:
		raise SystemExit(f"Не удалось прочитать {path}: {exc}") from exc

	if not rows:
		raise SystemExit(f"В файле нет корректных JSON-объектов: {path}")
	return pd.DataFrame(rows), invalid_lines


def clean_data(raw: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
	"""Normalize numeric fields and return RUB listings and usable price/m² rows."""
	data = raw.copy()
	for column in NUMERIC_COLUMNS:
		if column not in data:
			data[column] = np.nan
		data[column] = pd.to_numeric(data[column], errors="coerce")
		data[column] = data[column].replace([np.inf, -np.inf], np.nan)

	for column in ("currency", "highway", "locality", "seller_type", "object_type"):
		if column not in data:
			data[column] = pd.NA
		data[column] = data[column].astype("string").str.strip().replace("", pd.NA)

	# asking_price_rub is the dataset's RUB price field; a missing currency label
	# does not invalidate it, while an explicitly non-RUB record is excluded.
	currency = data["currency"].fillna("RUB").str.upper()
	rub = data.loc[currency.eq("RUB")].copy()
	rub = rub.loc[rub["asking_price_rub"].gt(0)].copy()
	rub["price_mln_rub"] = rub["asking_price_rub"] / 1_000_000

	usable = rub.loc[rub["house_area_m2"].gt(0)].copy()
	usable["price_per_m2_calc"] = (
		usable["asking_price_rub"] / usable["house_area_m2"]
	)
	usable["price_per_m2_mln"] = usable["price_per_m2_calc"] / 1_000_000
	advertised = usable["price_per_m2_advertised"].gt(0)
	usable["advertised_unit_price_diff_pct"] = np.nan
	usable.loc[advertised, "advertised_unit_price_diff_pct"] = (
		(usable.loc[advertised, "price_per_m2_advertised"]
		 - usable.loc[advertised, "price_per_m2_calc"]).abs()
		/ usable.loc[advertised, "price_per_m2_calc"]
		* 100
	)
	return rub, usable


def describe_metric(series: pd.Series) -> pd.DataFrame:
	"""Return a compact descriptive-statistics table for a numeric series."""
	values = series.dropna()
	if values.empty:
		return pd.DataFrame(columns=["metric", "value"])
	metrics = {
		"Количество": values.size,
		"Среднее": values.mean(),
		"Медиана": values.median(),
		"Стандартное отклонение": values.std(),
		"Минимум": values.min(),
		"10-й перцентиль": values.quantile(0.10),
		"25-й перцентиль": values.quantile(0.25),
		"75-й перцентиль": values.quantile(0.75),
		"90-й перцентиль": values.quantile(0.90),
		"95-й перцентиль": values.quantile(0.95),
		"Максимум": values.max(),
	}
	return pd.DataFrame({"metric": metrics.keys(), "value": metrics.values()})


def grouped_prices(data: pd.DataFrame, column: str, min_count: int = 5) -> pd.DataFrame:
	"""Summarize prices by a categorical field, omitting very small groups."""
	if column not in data:
		return pd.DataFrame(columns=["category", "count", "median_price_rub", "mean_price_rub"])
	subset = data.loc[data[column].notna(), [column, "asking_price_rub"]]
	result = (
		subset.groupby(column, observed=True)["asking_price_rub"]
		.agg(count="size", median_price_rub="median", mean_price_rub="mean")
		.reset_index()
		.rename(columns={column: "category"})
	)
	return result.loc[result["count"].ge(min_count)].sort_values(
		["count", "median_price_rub"], ascending=[False, True]
	)


def find_iqr_outliers(data: pd.DataFrame, column: str, metric_label: str) -> pd.DataFrame:
	"""Return Tukey 1.5×IQR outliers for one numeric metric."""
	values = data[column].dropna()
	if values.size < 4:
		return pd.DataFrame()
	q1, q3 = values.quantile([0.25, 0.75])
	iqr = q3 - q1
	lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
	mask = data[column].lt(lower) | data[column].gt(upper)
	columns = [name for name in (
		"listing_id", "listing_url", "locality", "highway", "asking_price_rub",
		"house_area_m2", "price_per_m2_calc", "land_area_sotka",
	) if name in data.columns]
	result = data.loc[mask, columns].copy()
	result["outlier_metric"] = metric_label
	result["iqr_lower_bound"] = lower
	result["iqr_upper_bound"] = upper
	return result


def markdown_table(frame: pd.DataFrame, max_rows: int = 20) -> str:
	"""Build a Markdown table without relying on the optional tabulate package."""
	if frame.empty:
		return "_Нет данных._"
	shown = frame.head(max_rows).copy()
	headers = [str(name) for name in shown.columns]

	def display(value: Any) -> str:
		if pd.isna(value):
			return "—"
		if isinstance(value, (float, np.floating)):
			return f"{value:,.2f}".replace(",", " ")
		return str(value).replace("|", "\\|").replace("\n", " ")

	lines = ["| " + " | ".join(headers) + " |",
			 "| " + " | ".join("---" for _ in headers) + " |"]
	for row in shown.itertuples(index=False, name=None):
		lines.append("| " + " | ".join(display(value) for value in row) + " |")
	if len(frame) > max_rows:
		lines.append(f"\nПоказаны первые {max_rows} из {len(frame)} строк.")
	return "\n".join(lines)


def save_csv(frame: pd.DataFrame, path: Path) -> None:
	frame.to_csv(path, index=False, encoding="utf-8-sig")


def save_charts(prices: pd.DataFrame, usable: pd.DataFrame,
				tags: pd.DataFrame, highway_summary: pd.DataFrame,
				missing: pd.DataFrame, figure_dir: Path) -> list[Path]:
	figure_dir.mkdir(parents=True, exist_ok=True)
	plt.rcParams.update({
		"font.family": "sans-serif",
		"font.sans-serif": ["Arial", "DejaVu Sans"],
		"axes.unicode_minus": False,
		"figure.dpi": 120,
	})
	outputs: list[Path] = []

	# Extreme values remain in tables/statistics; charts clip at the 99th percentile.
	price_values = prices["price_mln_rub"].dropna()
	if not price_values.empty:
		cap = price_values.quantile(0.99)
		visible = price_values.loc[price_values.le(cap)]
		fig, ax = plt.subplots(figsize=(10, 6))
		ax.hist(visible, bins=min(40, max(10, int(math.sqrt(len(visible))))),
				color="#2563eb", edgecolor="white")
		ax.set(title="Распределение цен предложения",
			   xlabel="Цена, млн ₽ (обрезано на 99-м перцентиле)", ylabel="Количество объявлений")
		ax.grid(axis="y", alpha=0.25)
		fig.tight_layout()
		path = figure_dir / "price_distribution.png"
		fig.savefig(path, dpi=180)
		plt.close(fig)
		outputs.append(path)

	unit_values = usable["price_per_m2_calc"].dropna() / 1_000
	if not unit_values.empty:
		cap = unit_values.quantile(0.99)
		visible = unit_values.loc[unit_values.le(cap)]
		fig, ax = plt.subplots(figsize=(10, 6))
		ax.hist(visible, bins=min(40, max(10, int(math.sqrt(len(visible))))),
				color="#0f766e", edgecolor="white")
		ax.set(title="Распределение цены за квадратный метр",
			   xlabel="Цена, тыс. ₽/м² (обрезано на 99-м перцентиле)",
			   ylabel="Количество объявлений")
		ax.grid(axis="y", alpha=0.25)
		fig.tight_layout()
		path = figure_dir / "price_per_m2_distribution.png"
		fig.savefig(path, dpi=180)
		plt.close(fig)
		outputs.append(path)

	if not usable.empty:
		x_cap = usable["house_area_m2"].quantile(0.99)
		y_cap = usable["asking_price_rub"].quantile(0.99)
		visible = usable.loc[
			usable["house_area_m2"].le(x_cap)
			& usable["asking_price_rub"].le(y_cap)
		]
		fig, ax = plt.subplots(figsize=(10, 6))
		ax.scatter(visible["house_area_m2"], visible["price_mln_rub"],
				   alpha=0.55, s=24, color="#7c3aed", edgecolors="none")
		ax.set(title="Цена предложения и площадь дома",
			   xlabel="Площадь дома, м²", ylabel="Цена, млн ₽")
		ax.grid(alpha=0.25)
		fig.tight_layout()
		path = figure_dir / "price_vs_area.png"
		fig.savefig(path, dpi=180)
		plt.close(fig)
		outputs.append(path)

	if not highway_summary.empty:
		chart_data = highway_summary.head(12).sort_values("median_price_rub")
		fig, ax = plt.subplots(figsize=(11, max(5, len(chart_data) * 0.42)))
		ax.barh(chart_data["category"], chart_data["median_price_rub"] / 1_000_000,
				color="#ea580c")
		ax.set(title="Медианная цена по шоссе (группы от 5 объявлений)",
			   xlabel="Медианная цена, млн ₽", ylabel="Шоссе")
		ax.grid(axis="x", alpha=0.25)
		fig.tight_layout()
		path = figure_dir / "median_price_by_highway.png"
		fig.savefig(path, dpi=180)
		plt.close(fig)
		outputs.append(path)

	if not tags.empty:
		chart_data = tags.head(15).sort_values("listing_count")
		fig, ax = plt.subplots(figsize=(10, max(5, len(chart_data) * 0.36)))
		ax.barh(chart_data["tag"], chart_data["listing_count"], color="#0891b2")
		ax.set(title="Частые заявленные характеристики и окружение",
			   xlabel="Объявлений с меткой", ylabel="Метка")
		ax.grid(axis="x", alpha=0.25)
		fig.tight_layout()
		path = figure_dir / "claimed_tags.png"
		fig.savefig(path, dpi=180)
		plt.close(fig)
		outputs.append(path)

	if not missing.empty:
		chart_data = missing.head(20).sort_values("missing_pct")
		fig, ax = plt.subplots(figsize=(10, max(5, len(chart_data) * 0.36)))
		ax.barh(chart_data["column"], chart_data["missing_pct"], color="#64748b")
		ax.set(title="Доля пропусков по полям", xlabel="Пропуски, %", ylabel="Поле")
		ax.set_xlim(0, 100)
		ax.grid(axis="x", alpha=0.25)
		fig.tight_layout()
		path = figure_dir / "missing_values.png"
		fig.savefig(path, dpi=180)
		plt.close(fig)
		outputs.append(path)

	return outputs


def run_analysis(input_path: Path, output_dir: Path) -> None:
	raw, invalid_lines = read_jsonl(input_path)
	prices, usable = clean_data(raw)
	if prices.empty:
		raise SystemExit("Не найдено объявлений с положительной ценой в рублях.")

	output_dir.mkdir(parents=True, exist_ok=True)
	figure_dir = output_dir / "figures"
	missing = pd.DataFrame({
		"column": raw.columns,
		"missing_count": raw.isna().sum().values,
		"missing_pct": (raw.isna().mean().values * 100),
	}).sort_values(["missing_pct", "column"], ascending=[False, True])
	price_bins = (
		pd.cut(prices["asking_price_rub"], bins=PRICE_BINS, labels=PRICE_BIN_LABELS,
			   right=False, include_lowest=True)
		.value_counts(sort=False)
		.rename_axis("price_range")
		.reset_index(name="listing_count")
	)
	price_bins["share_pct"] = price_bins["listing_count"] / len(prices) * 100

	grouped = {
		field: grouped_prices(prices, field)
		for field in ("highway", "locality", "seller_type", "object_type")
	}
	tag_rows: list[dict[str, Any]] = []
	if "tags_seller_claimed" in raw:
		tag_values = raw["tags_seller_claimed"].explode().dropna().astype(str).str.strip()
		tag_values = tag_values.loc[tag_values.ne("")]
		if not tag_values.empty:
			tag_rows = [
				{"tag": str(tag), "listing_count": int(count),
				 "share_pct": float(count / len(raw) * 100)}
				for tag, count in tag_values.value_counts().items()
			]
	tags = pd.DataFrame(tag_rows, columns=["tag", "listing_count", "share_pct"])

	outliers = pd.concat([
		find_iqr_outliers(prices, "asking_price_rub", "Цена предложения, ₽"),
		find_iqr_outliers(usable, "price_per_m2_calc", "Цена за м², ₽/м²"),
	], ignore_index=True)
	if outliers.empty:
		outliers = pd.DataFrame(columns=[
			"listing_id", "listing_url", "locality", "highway", "asking_price_rub",
			"house_area_m2", "price_per_m2_calc", "land_area_sotka",
			"outlier_metric", "iqr_lower_bound", "iqr_upper_bound",
		])

	numeric = usable[[column for column in NUMERIC_COLUMNS
					  if column in usable and usable[column].notna().sum() >= 2]].copy()
	numeric["price_per_m2_calc"] = usable["price_per_m2_calc"]
	correlations = numeric.corr(method="spearman", numeric_only=True)
	correlations.index.name = "feature"

	distributions = pd.concat([
		describe_metric(prices["asking_price_rub"]).assign(metric_group="Цена предложения, ₽"),
		describe_metric(usable["price_per_m2_calc"]).assign(metric_group="Цена за м², ₽/м²"),
		describe_metric(usable["house_area_m2"]).assign(metric_group="Площадь дома, м²"),
		describe_metric(usable["land_area_sotka"]).assign(metric_group="Площадь участка, соток"),
		describe_metric(usable["distance_to_mkad_km"]).assign(metric_group="Расстояние от МКАД, км"),
	], ignore_index=True)[["metric_group", "metric", "value"]]
	report_stats = (
		distributions.pivot(index="metric_group", columns="metric", values="value")
		.reindex(columns=["Количество", "Медиана", "Среднее", "Минимум", "Максимум"])
		.rename_axis(None, axis="columns")
		.reset_index()
		.rename(columns={"metric_group": "Показатель"})
	)

	files: dict[str, pd.DataFrame] = {
		"price_distribution.csv": price_bins,
		"numeric_summary.csv": distributions,
		"missing_values.csv": missing,
		"claimed_tags.csv": tags,
		"outliers_iqr.csv": outliers,
		"spearman_correlations.csv": correlations.reset_index(),
	}
	for field, frame in grouped.items():
		files[f"price_by_{field}.csv"] = frame
	for filename, frame in files.items():
		save_csv(frame, output_dir / filename)

	highway_summary = grouped["highway"]
	charts = save_charts(prices, usable, tags, highway_summary, missing, figure_dir)

	price_stats = prices["asking_price_rub"].describe(
		percentiles=[0.10, 0.25, 0.5, 0.75, 0.90, 0.95]
	)
	unit_stats = usable["price_per_m2_calc"].describe(
		percentiles=[0.10, 0.25, 0.5, 0.75, 0.90, 0.95]
	) if not usable.empty else pd.Series(dtype=float)
	duplicate_ids = int(raw["listing_id"].duplicated().sum()) if "listing_id" in raw else 0
	mismatch = usable["advertised_unit_price_diff_pct"].dropna()
	mismatch_over_5 = int(mismatch.gt(5).sum())

	report = [
		"# EDA: объявления о недвижимости Домклик",
		"",
		f"Источник: `{input_path}`",
		f"Всего корректных записей: **{len(raw):,}**; некорректных строк JSON: **{invalid_lines}**.",
		f"Объявлений с положительной ценой в RUB: **{len(prices):,}**; "
		f"с валидной площадью дома для расчёта цены за м²: **{len(usable):,}**.",
		f"Повторяющихся listing_id сверх первого: **{duplicate_ids}**.",
		"",
		"## Цена предложения",
		f"Медиана: **{price_stats.get('50%', float('nan')):,.0f} ₽**; "
		f"среднее: **{price_stats.get('mean', float('nan')):,.0f} ₽**; "
		f"диапазон: **{price_stats.get('min', float('nan')):,.0f}–"
		f"{price_stats.get('max', float('nan')):,.0f} ₽**."
		.replace(",", " "),
		"",
		"Распределение по ценовым диапазонам (до построения графиков данные не удаляются):",
		markdown_table(price_bins.assign(
			share_pct=price_bins["share_pct"].round(1)
		).rename(columns={"price_range": "Диапазон", "listing_count": "Объявлений",
						  "share_pct": "Доля, %"})),
		"",
		"## Цена за м² и площади",
		(f"Медиана расчётной цены за м²: **{unit_stats.get('50%', float('nan')):,.0f} ₽/м²**; "
		 f"медианная площадь дома: **{usable['house_area_m2'].median():,.1f} м²**."
		 .replace(",", " ") if not usable.empty else "Нет строк с корректной площадью дома."),
		f"У объявлений с обеими ценами расхождение рекламной и пересчитанной цены за м² "
		f"более 5% найдено в **{mismatch_over_5:,}** случаях из **{len(mismatch):,}**.",
		"",
		"## Сводка по объявлениям",
		markdown_table(report_stats),
		"",
		"## По шоссе (группы от 5 объявлений)",
		markdown_table(highway_summary.head(15).rename(columns={
			"category": "Шоссе", "count": "Объявлений",
			"median_price_rub": "Медиана цены, ₽", "mean_price_rub": "Средняя цена, ₽"
		})),
		"",
		"## По типу продавца",
		markdown_table(grouped["seller_type"].rename(columns={
			"category": "Тип продавца", "count": "Объявлений",
			"median_price_rub": "Медиана цены, ₽", "mean_price_rub": "Средняя цена, ₽"
		})),
		"",
		"## По типу объекта",
		markdown_table(grouped["object_type"].rename(columns={
			"category": "Тип объекта", "count": "Объявлений",
			"median_price_rub": "Медиана цены, ₽", "mean_price_rub": "Средняя цена, ₽"
		})),
		"",
		"## Частые населённые пункты (группы от 5 объявлений)",
		markdown_table(grouped["locality"].head(15).rename(columns={
			"category": "Населённый пункт", "count": "Объявлений",
			"median_price_rub": "Медиана цены, ₽", "mean_price_rub": "Средняя цена, ₽"
		})),
		"",
		"## Частые заявленные метки",
		markdown_table(tags.head(15).rename(columns={
			"tag": "Метка", "listing_count": "Объявлений", "share_pct": "Доля, %"
		}).assign(**{"Доля, %": tags.head(15)["share_pct"].round(1)})),
		"",
		"## Качество данных и ограничения",
		f"- По правилу 1.5×IQR отмечено **{len(outliers):,}** строк-меток выбросов "
		"(одна карточка может попасть в обе метрики). Это кандидаты на проверку, а не автоматически ошибки.",
		"- Для цены за м² используется цена предложения / площадь дома; это не цена сделки.",
		"- Заявленные продавцом метки не подтверждают наличие коммуникаций или характеристик.",
		"- Гистограммы и scatter ограничены 99-м перцентилем для читаемости; полные значения сохранены в таблицах и статистике.",
		"- Групповые медианы не являются оценкой рыночной стоимости; состав объявлений и заполненность полей могут различаться.",
		"",
		"## Файлы",
		*[f"- `{filename}`" for filename in files],
		*[f"- `figures/{path.name}`" for path in charts],
	]
	(output_dir / "report.md").write_text("\n".join(report) + "\n", encoding="utf-8")

	print(f"Записей в источнике: {len(raw):,} (битых строк: {invalid_lines})".replace(",", " "))
	print(f"Цена RUB > 0: {len(prices):,}; цена за м²: {len(usable):,}".replace(",", " "))
	print(f"Медианная цена: {prices['asking_price_rub'].median():,.0f} ₽".replace(",", " "))
	if not usable.empty:
		print(f"Медианная цена за м²: {usable['price_per_m2_calc'].median():,.0f} ₽/м²".replace(",", " "))
	print(f"Отчёт и результаты сохранены в: {output_dir}")


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(
		description="Исследование распределения цен и характеристик объявлений из objects.jsonl."
	)
	parser.add_argument(
		"--input", type=Path, default=default_input_path(),
		help="Путь к objects.jsonl (по умолчанию поиск в data/dataset/).",
	)
	parser.add_argument(
		"--output-dir", type=Path, default=Path(__file__).resolve().parent / "reports" / "eda",
		help="Каталог для отчёта, CSV и figures/ (по умолчанию reports/eda).",
	)
	parser.add_argument("--verbose", action="store_true", help="Показывать предупреждения и подробные сообщения.")
	return parser.parse_args()


def main() -> None:
	args = parse_args()
	logging.basicConfig(
		level=logging.INFO if args.verbose else logging.WARNING,
		format="%(levelname)s: %(message)s",
	)
	run_analysis(args.input.expanduser(), args.output_dir.expanduser())


if __name__ == "__main__":
	main()
