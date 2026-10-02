#!/usr/bin/env python3
"""Extract modeling features from the Domclick objects.jsonl dataset.

Features: house_area_m2, land_area_sotka.
Target:   asking_price_rub (asking price, not a confirmed sale price).

To add more features later, extend FEATURE_COLUMNS — validation, filtering,
summaries, correlations and the report all derive from that tuple.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


LOG = logging.getLogger("domclick.features")
IDENTIFIER_COLUMNS = ("listing_id",)
FEATURE_COLUMNS = ("house_area_m2", "land_area_sotka")
TARGET_COLUMN = "asking_price_rub"


def default_input_path() -> Path:
    """Find the generated dataset in the usual project output locations."""
    root = Path(__file__).resolve().parent
    candidates = (root / "data" / "dataset" / "objects.jsonl", )
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return candidates[0]


def default_features_path() -> Path:
    """Where the modeling table goes: data/features/, next to the source data."""
    return Path(__file__).resolve().parent / "data" / "features" / "features.csv"


def default_report_dir() -> Path:
    """Where the report and summaries go: reports/features/, separate from the data."""
    return Path(__file__).resolve().parent / "reports" / "features"


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


def to_numeric(data: pd.DataFrame, column: str) -> pd.Series:
    """Convert a column to float, turning blanks and infinities into NaN."""
    if column not in data:
        data[column] = np.nan
    values = pd.to_numeric(data[column], errors="coerce")
    return values.replace([np.inf, -np.inf], np.nan)


def extract_features(raw: pd.DataFrame, fill_missing: str) -> tuple[pd.DataFrame, dict[str, int]]:
    """Build the feature table and count how many rows each check removed."""
    missing_columns = [
        column for column in (*IDENTIFIER_COLUMNS, *FEATURE_COLUMNS, TARGET_COLUMN)
        if column not in raw.columns
    ]
    if missing_columns:
        raise SystemExit("В источнике нет обязательных полей: " + ", ".join(missing_columns))

    data = raw.copy()
    for column in (*FEATURE_COLUMNS, TARGET_COLUMN):
        data[column] = to_numeric(data, column)

    rejected: dict[str, int] = {}

    # The target must be a real positive amount; zero or negative price is not usable.
    invalid_target = data[TARGET_COLUMN].isna() | data[TARGET_COLUMN].le(0)
    rejected["target_missing_or_not_positive"] = int(invalid_target.sum())
    data = data.loc[~invalid_target].copy()

    # Areas are physical quantities: negative values are data errors, zero is useless.
    invalid_features = pd.Series(False, index=data.index)
    for column in FEATURE_COLUMNS:
        negative = data[column].lt(0)
        zero = data[column].eq(0)
        rejected[f"{column}_negative"] = int(negative.sum())
        rejected[f"{column}_zero"] = int(zero.sum())
        invalid_features |= negative | zero
    for column in FEATURE_COLUMNS:
        rejected[f"{column}_missing"] = int(data[column].isna().sum())
    data = data.loc[~invalid_features].copy()

    if fill_missing == "median":
        for column in FEATURE_COLUMNS:
            data[column] = data[column].fillna(data[column].median())
    else:
        complete = data[list(FEATURE_COLUMNS)].notna().all(axis=1)
        rejected["row_dropped_for_missing_feature"] = int((~complete).sum())
        data = data.loc[complete].copy()

    columns = [*IDENTIFIER_COLUMNS, *FEATURE_COLUMNS, TARGET_COLUMN]
    return data[columns].reset_index(drop=True), rejected


def summary_table(features: pd.DataFrame) -> pd.DataFrame:
    """Describe feature and target columns for the report and CSV output."""
    rows: list[dict[str, Any]] = []
    for column in (*FEATURE_COLUMNS, TARGET_COLUMN):
        values = features[column].dropna()
        rows.append({
            "column": column,
            "role": "target" if column == TARGET_COLUMN else "feature",
            "count": int(values.size),
            "missing": int(features[column].isna().sum()),
            "mean": values.mean(),
            "median": values.median(),
            "std": values.std(),
            "min": values.min(),
            "max": values.max(),
            "skew": values.skew(),
        })
    return pd.DataFrame(rows)


def markdown_table(frame: pd.DataFrame) -> str:
    """Build a small Markdown table without relying on the optional tabulate package."""
    if frame.empty:
        return "_Нет данных._"
    headers = [str(name) for name in frame.columns]

    def display(value: Any) -> str:
        if value is None or (not isinstance(value, str) and pd.isna(value)):
            return "—"
        if isinstance(value, str):
            return value
        if isinstance(value, (int, np.integer)):
            return f"{value:,}".replace(",", " ")
        return f"{value:,.2f}".replace(",", " ")

    lines = ["| " + " | ".join(headers) + " |",
             "| " + " | ".join("---" for _ in headers) + " |"]
    for row in frame.itertuples(index=False, name=None):
        lines.append("| " + " | ".join(display(value) for value in row) + " |")
    return "\n".join(lines)


def run(input_path: Path, output_dir: Path, fill_missing: str,
        features_path: Path | None = None) -> None:
    raw, invalid_lines = read_jsonl(input_path)
    features, rejected = extract_features(raw, fill_missing)
    if features.empty:
        raise SystemExit("После проверок не осталось ни одной строки.")

    output_dir.mkdir(parents=True, exist_ok=True)
    table_path = features_path or default_features_path()
    table_path.parent.mkdir(parents=True, exist_ok=True)
    features.to_csv(table_path, index=False, encoding="utf-8-sig")
    summary = summary_table(features)
    summary.to_csv(output_dir / "feature_summary.csv", index=False, encoding="utf-8-sig")

    correlation = features[[*FEATURE_COLUMNS, TARGET_COLUMN]].corr(method="spearman")
    correlation.index.name = "feature"
    correlation.reset_index().to_csv(
        output_dir / "feature_correlations.csv", index=False, encoding="utf-8-sig"
    )
    target_correlations = (
        correlation[TARGET_COLUMN].drop(labels=[TARGET_COLUMN])
        .rename("spearman_with_target").reset_index()
    )

    removed = len(raw) - invalid_lines - len(features)
    report = [
        "# Признаки для модели цены",
        "",
        f"Источник: `{input_path}`",
        f"Признаки: **{', '.join(FEATURE_COLUMNS)}**; целевая переменная: **{TARGET_COLUMN}**.",
        f"Записей в источнике: **{len(raw):,}**; битых строк JSON: **{invalid_lines}**.",
        f"Осталось в выборке: **{len(features):,}**; удалено строк: **{removed:,}** "
        f"({removed / len(raw) * 100:.1f} % от источника).",
        "",
        "## Сводка по столбцам",
        markdown_table(summary),
        "",
        "## Корреляция признаков с ценой (Спирмен)",
        markdown_table(target_correlations),
        "",
        "## Проверки качества данных",
        "Число строк по каждой проверке; одна строка может попасть в несколько проверок.",
        markdown_table(pd.DataFrame(
            [{"Проверка": key, "Строк": value} for key, value in rejected.items()]
        )),
        "",
        "## Колонки файла",
        *[f"- `{column}`" for column in features.columns],
        "",
        "## Файлы",
        f"- Таблица признаков: `{table_path}`",
        f"- Сводка по столбцам: `{output_dir / 'feature_summary.csv'}`",
        f"- Корреляции: `{output_dir / 'feature_correlations.csv'}`",
        f"- Отчёт: `{output_dir / 'report.md'}`",
        "",
        "## Ограничения",
        "- `asking_price_rub` — цена предложения, а не подтверждённая цена сделки.",
        "- Площадь из карточки может расходиться с документами и с текстом объявления.",
        "- Медиана и среднее цены сильно расходятся, `skew` положительный:",
        "  распределение цены скошено, для линейных моделей может понадобиться логарифм цели.",
        "- Признаков всего два; локация, материалы и транспорт не учтены и уходят в остаток.",
    ]
    (output_dir / "report.md").write_text("\n".join(report) + "\n", encoding="utf-8")

    print(f"Записей в источнике: {len(raw):,} (битых строк: {invalid_lines})".replace(",", " "))
    print(f"Строк с признаками: {len(features):,}".replace(",", " "))
    print(f"Медианная цена: {features[TARGET_COLUMN].median():,.0f} ₽".replace(",", " "))
    print(f"Таблица признаков: {table_path}")
    print(f"Отчёт и сводки: {output_dir / 'report.md'}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Извлечение признаков (площадь дома, площадь участка) и цены из objects.jsonl."
    )
    parser.add_argument(
        "--input", type=Path, default=default_input_path(),
        help="Путь к objects.jsonl (по умолчанию поиск в data/dataset/).",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=default_report_dir(),
        help="Каталог для отчёта и сводок (по умолчанию reports/features).",
    )
    parser.add_argument(
        "--features-file", type=Path, default=default_features_path(),
        help="Куда записать таблицу признаков (по умолчанию data/features/features.csv).",
    )
    parser.add_argument(
        "--fill-missing", choices=("drop", "median"), default="drop",
        help="Что делать со строками без площади: 'drop' (по умолчанию) или 'median'.",
    )
    parser.add_argument("--verbose", action="store_true", help="Показывать предупреждения и подробные сообщения.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(levelname)s: %(message)s",
    )
    run(
        args.input.expanduser(),
        args.output_dir.expanduser(),
        args.fill_missing,
        args.features_file.expanduser(),
    )


if __name__ == "__main__":
    main()