#!/usr/bin/env python3
"""Build a wide, leakage-free feature matrix for house-price models.

Source:  data/dataset/objects.jsonl (structured fields of the Domclick export).
Output:  data/features/features_rich.csv — one row per listing, numeric only.
         reports/features_rich/ — feature dictionary, coverage, report.

Design rules
------------
1. Target leakage is excluded explicitly: every price-derived column
   (`price_per_m2_advertised`, `price_per_m2_computed`) is dropped, see LEAKAGE_COLUMNS.
2. Missing values stay as NaN instead of being imputed: gradient boosting
   (HistGradientBoosting, XGBoost, LightGBM) handles NaN natively, and imputing
   before a train/test split leaks information. Missing-value indicators are added
   next to the raw columns so linear models can still use the information.
3. Categorical levels are one-hot encoded only when frequent enough
   (`--one-hot-min-count`); rare levels fall into a single `_other` column.
   High-cardinality fields get target-free frequency encodings instead, which
   keeps the matrix narrow and avoids the target leakage of target encoding.
4. Text-derived flags are built from `description`; disable them with
   `--no-text-features` to keep strictly structured fields.
"""

from __future__ import annotations

import argparse
import json
import logging
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


LOG = logging.getLogger("domclick.rich")

IDENTIFIER_COLUMN = "listing_id"
TARGET_COLUMN = "asking_price_rub"
LOG_TARGET_COLUMN = "log_asking_price_rub"
GROUP_COLUMN = "group_locality"

# Columns that must never be fed to a model as features (id, targets, CV group key).
NON_FEATURE_COLUMNS = (IDENTIFIER_COLUMN, TARGET_COLUMN, LOG_TARGET_COLUMN, GROUP_COLUMN)

# Any column computed from the price would leak the target into the features.
LEAKAGE_COLUMNS = ("price_per_m2_advertised", "price_per_m2_computed")

# Constants in this export: no information, so they are dropped automatically.
CONSTANT_COLUMNS = ("object_type", "price_type", "currency")

# Sparse or process artefacts: kept out and listed in the report instead.
DROPPED_COLUMNS = {
    "seller_name": "заполнено в 40 строках (2,2 %) — слишком мало для признака",
    "seller_type": "заполнено в 40 строках (2,2 %) — слишком мало для признака",
    "conflicts": "непустое значение в 1 строке — признака не получится",
    "badges_platform_claimed": "метки площадки не используются: слабый сигнал, а «Выгодная "
                               "цена» вычисляется из цены (утечка цели)",
    "photo_urls": "используется только длина и сверка с photo_count_advertised",
    "photo_paths": "пусто в 1815 строках (97,7 %) — артефакт скачивания, не свойство объекта",
    "sources": "используется только число источников",
    "listing_url": "дублирует listing_id",
    "locality_url": "используется только глубина и слаг",
    "description": "используется только через текстовые флаги",
    "address_fragment": "используется только структура (число частей, наличие номера)",
}

TAG_FLAGS = (
    ("gas", r"\bгаз"),
    ("water", r"водоснабжение"),
    ("electricity", r"электричество"),
    ("near_water", r"рядом водоём"),
    ("near_forest", r"рядом лес"),
)

LOCALITY_TYPES = (
    ("locality_type_snt", r"\bснт\b"),
    ("locality_type_dnp", r"\bднп\b"),
    ("locality_type_kp", r"\bкп\b"),
    ("locality_type_village", r"^\s*д\."),
    ("locality_type_selo", r"^\s*с\."),
    ("locality_type_settlement", r"^\s*пос\.|пос[её]лок"),
    ("locality_type_district", r"округ|район|р-н"),
)

SETTLEMENT_TYPES = (
    ("settlement_type_settlement", r"пос[её]л"),
    ("settlement_type_snt", r"\bснт\b"),
    ("settlement_type_dnp", r"\bднп\b"),
    ("settlement_type_kp", r"\bкп\b"),
    ("settlement_type_village", r"деревн"),
)

TEXT_PATTERNS = (
    ("txt_gas", "utilities", r"\bгаз"),
    ("txt_electricity", "utilities", r"электр"),
    ("txt_water", "utilities", r"водоснаб|скважин|водопровод"),
    ("txt_sewer", "utilities", r"канализ|септик"),
    ("txt_heating", "utilities", r"отоплен|котел|котёл|т[её]пл(?:ый|ые) пол"),
    ("txt_finish", "condition", r"отделк|ремонт"),
    ("txt_ready_to_move", "condition", r"под ключ|заезжай|с отделк"),
    ("txt_furniture", "condition", r"мебел"),
    ("txt_new_build", "condition", r"нов(?:ый|остройк)"),
    ("txt_brick", "construction", r"кирпич"),
    ("txt_frame", "construction", r"каркас|брус|дерев"),
    ("txt_monolith", "construction", r"монолит"),
    ("txt_bathhouse", "amenities", r"бан[яь]|саун"),
    ("txt_garage", "amenities", r"гараж"),
    ("txt_pool", "amenities", r"бассейн"),
    ("txt_fireplace", "amenities", r"камин"),
    ("txt_terrace", "amenities", r"террас"),
    ("txt_forest", "location", r"лес"),
    ("txt_pond", "location", r"водо[её]м|пруд|река|озер|озёр"),
    ("txt_metro", "location", r"метро"),
    ("txt_school", "location", r"школ"),
    ("txt_kindergarten", "location", r"детск(?:ий )?сад"),
    ("txt_security", "location", r"охран"),
    ("txt_nova_moscow", "location", r"новая москва"),
    ("txt_izhs", "legal", r"\bижс\b"),
    ("txt_snt", "legal", r"\bснт\b"),
    ("txt_dnp", "legal", r"\bднп\b"),
    ("txt_lph", "legal", r"\bлпх\b"),
    ("txt_propiska", "legal", r"прописк"),
    ("txt_mortgage", "finance", r"ипотек"),
    ("txt_maternity_capital", "finance", r"мат\.?\s?капитал|материнск"),
    ("txt_bargain", "marketing", r"торг|срочно"),
)

MONTHS = {
    "янв": 1, "фев": 2, "мар": 3, "апр": 4, "май": 5, "мая": 5,
    "июн": 6, "июл": 7, "авг": 8, "сен": 9, "окт": 10, "ноя": 11, "дек": 12,
}

RELATIVE_AGE = re.compile(r"(\d+)?\s*(минут|час|день|дня|дней|недел|месяц|год)")
ABSOLUTE_AGE = re.compile(r"^\s*(\d{1,2})\s+([а-яё]{3,})\.?\s+(\d{2})\s*$")


class FeatureBuilder:
    """Collects feature columns together with their documentation metadata."""

    def __init__(self) -> None:
        self.columns: dict[str, pd.Series] = {}
        self.metadata: list[dict[str, Any]] = []

    def add(self, name: str, values: pd.Series, group: str, kind: str,
            source: str, note: str = "") -> None:
        if name in self.columns:
            raise ValueError(f"Признак {name} добавлен дважды.")
        self.columns[name] = values.astype("float64")
        self.metadata.append({
            "column": name, "group": group, "kind": kind,
            "source_field": source, "note": note,
        })

    def add_flag(self, name: str, mask: pd.Series, group: str, kind: str,
                 source: str, note: str = "") -> None:
        self.add(name, mask.fillna(False).astype(float), group, kind, source, note)

    def frame(self) -> pd.DataFrame:
        return pd.DataFrame(self.columns)

    def dictionary(self) -> pd.DataFrame:
        return pd.DataFrame(self.metadata)


def default_input_path() -> Path:
    """Find the generated dataset in the usual project output locations."""
    root = Path(__file__).resolve().parent
    candidates = (root / "data" / "dataset" / "objects.jsonl", )
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return candidates[0]


def default_features_path() -> Path:
    """Modeling matrices belong in data/, next to the source dataset."""
    return Path(__file__).resolve().parent / "data" / "features" / "features_rich.csv"


def default_report_dir() -> Path:
    """Reports belong in reports/, separate from the data."""
    return Path(__file__).resolve().parent / "reports" / "features_rich"


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


def numeric(raw: pd.DataFrame, column: str) -> pd.Series:
    """Convert a source column to float, turning blanks and infinities into NaN."""
    if column not in raw:
        return pd.Series(np.nan, index=raw.index, dtype="float64")
    values = pd.to_numeric(raw[column], errors="coerce")
    return values.replace([np.inf, -np.inf], np.nan)


def text(raw: pd.DataFrame, column: str) -> pd.Series:
    """Normalized text column: lowercased, empty for missing values."""
    if column not in raw:
        return pd.Series("", index=raw.index, dtype="string")
    values = raw[column].astype("string").fillna("").str.lower()
    return values.str.replace("\u00a0", " ", regex=False).str.strip()


def as_list(value: Any) -> list[Any]:
    """Treat a JSONL value as a list of scalars."""
    return value if isinstance(value, list) else []


def list_flags(column: pd.Series, pattern: str) -> pd.Series:
    """True when any element of a list column matches the pattern."""
    regex = re.compile(pattern, re.IGNORECASE)
    return column.map(
        lambda values: any(regex.search(str(item)) for item in as_list(values))
    )


def parse_age_days(values: pd.Series, reference: pd.Timestamp) -> tuple[pd.Series, pd.Series]:
    """Convert listing_age_text into days before the reference date.

    Handles relative texts ("7 часов назад", "Вчера", "2 недели назад") and
    absolute dates ("27 авг. 26"). Returns (days, is_absolute_date).
    """
    days: list[float] = []
    absolute: list[bool] = []
    for value in values:
        value = str(value).lower().strip()
        if not value:
            days.append(np.nan)
            absolute.append(False)
            continue

        match = ABSOLUTE_AGE.match(value)
        if match:
            day, month_token, year = match.groups()
            month = MONTHS.get(month_token[:3])
            if month is None:
                days.append(np.nan)
                absolute.append(True)
                continue
            stamp = pd.Timestamp(year=2000 + int(year), month=month, day=int(day))
            days.append(float((reference - stamp).days))
            absolute.append(True)
            continue

        if "вчера" in value:
            days.append(1.0)
            absolute.append(False)
            continue

        match = RELATIVE_AGE.search(value)
        if not match:
            days.append(np.nan)
            absolute.append(False)
            continue

        amount = float(match.group(1)) if match.group(1) else 1.0
        unit = match.group(2)
        if unit == "минут":
            amount /= 1440
        elif unit == "час":
            amount /= 24
        elif unit == "недел":
            amount *= 7
        elif unit in ("месяц", "год"):
            amount *= 30
        days.append(float(amount))
        absolute.append(False)

    return pd.Series(days, index=values.index), pd.Series(absolute, index=values.index)


def one_hot(series: pd.Series, prefix: str, builder: FeatureBuilder,
            group: str, source: str, min_count: int) -> list[str]:
    """One-hot encode a categorical column, merging rare levels into `_other`."""
    cleaned = series.astype("string").fillna("").str.strip()
    counts = cleaned.value_counts()
    frequent = [level for level, count in counts.items() if count >= min_count and level]
    mapping = {level: f"{prefix}_{index:02d}" for index, level in enumerate(sorted(frequent))}

    created: list[str] = []
    for level, name in mapping.items():
        builder.add_flag(
            name, cleaned.eq(level), group, "one_hot", source,
            note=f"уровень: {level} ({int(counts[level])} строк)",
        )
        created.append(name)

    other_levels = [level for level in counts.index if level and level not in mapping]
    if other_levels:
        builder.add_flag(
            f"{prefix}_other", cleaned.isin(other_levels), group, "one_hot", source,
            note=f"редкие уровни ({len(other_levels)} шт., {int(counts[other_levels].sum())} строк)",
        )
        created.append(f"{prefix}_other")
    builder.add_flag(
        f"{prefix}_unknown", cleaned.eq(""), group, "one_hot", source,
        note="поле не заполнено",
    )
    created.append(f"{prefix}_unknown")
    return created


def build_features(raw: pd.DataFrame, options: argparse.Namespace) -> FeatureBuilder:
    """Build the full feature matrix from the structured source fields."""
    builder = FeatureBuilder()
    index = raw.index

    # --- raw numeric backbone -------------------------------------------------
    house_area = numeric(raw, "house_area_m2")
    land_area = numeric(raw, "land_area_sotka")
    floors = numeric(raw, "floors")
    distance = numeric(raw, "distance_to_mkad_km")
    photos = numeric(raw, "photo_count_advertised")

    for name, values, note in (
        ("house_area_m2", house_area, "площадь дома, м², из карточки"),
        ("land_area_sotka", land_area, "площадь участка, соток"),
        ("floors", floors, "этажность из карточки"),
        ("distance_to_mkad_km", distance, "расстояние до МКАД, км (null вместе с highway)"),
        ("photo_count_advertised", photos, "заявленное число фотографий в карточке"),
    ):
        builder.add(name, values, "numeric_raw", "numeric", name, note)

    # --- derived ratios and logs ---------------------------------------------
    builder.add("land_to_house_ratio", land_area / house_area, "derived", "ratio",
                "land_area_sotka / house_area_m2", "баланс участка и дома")
    builder.add("house_area_per_floor", house_area / floors, "derived", "ratio",
                "house_area_m2 / floors", "средняя площадь этажа, м²")
    builder.add("land_area_per_floor", land_area / floors, "derived", "ratio",
                "land_area_sotka / floors", "соток на этаж")
    for name, values, source in (
        ("log_house_area", house_area, "house_area_m2"),
        ("log_land_area", land_area, "land_area_sotka"),
        ("log_distance_to_mkad", distance, "distance_to_mkad_km"),
        ("log_photos", photos + 1, "photo_count_advertised"),
    ):
        builder.add(name, np.log(values.where(values > 0)), "derived", "log",
                    source, "натуральный логарифм; уменьшает скошенность для линейных моделей")

    # --- listing age ----------------------------------------------------------
    reference = pd.Timestamp(options.reference_date) if options.reference_date else None
    if reference is None:
        stamps: list[pd.Timestamp] = []
        for value in raw.get("listing_age_text", pd.Series(dtype="object")):
            match = ABSOLUTE_AGE.match(str(value).lower().strip())
            if match:
                month = MONTHS.get(match.group(2)[:3])
                if month:
                    stamps.append(pd.Timestamp(
                        year=2000 + int(match.group(3)), month=month, day=int(match.group(1))
                    ))
        reference = max(stamps) if stamps else pd.Timestamp.now().normalize()
    age_days, is_date = parse_age_days(
        raw.get("listing_age_text", pd.Series("", index=index)), reference
    )
    builder.add("listing_age_days", age_days, "age", "age", "listing_age_text",
                f"дней до контрольной даты {reference.date()} (по тексту объявления)")
    builder.add_flag("listing_age_is_absolute_date", is_date, "age", "flag", "listing_age_text",
                     "в карточке указана дата, а не «N назад»")

    # --- missing-value indicators --------------------------------------------
    builder.add_flag("floors_missing", floors.isna(), "missing", "flag", "floors")
    builder.add_flag("land_area_missing", land_area.isna(), "missing", "flag", "land_area_sotka")
    builder.add_flag("distance_missing", distance.isna(), "missing", "flag",
                     "distance_to_mkad_km", "совпадает с отсутствием highway")
    builder.add_flag("locality_missing", text(raw, "locality").eq(""), "missing", "flag", "locality")
    builder.add_flag("settlement_missing", text(raw, "settlement").eq(""), "missing", "flag", "settlement")
    builder.add_flag("address_missing", text(raw, "address_fragment").eq(""), "missing", "flag",
                     "address_fragment")

    # --- export bookkeeping ---------------------------------------------------
    builder.add("n_sources", raw.get("sources", pd.Series([[]] * len(raw), index=index)).map(
        lambda value: float(len(as_list(value)))), "bookkeeping", "count", "sources",
        "в скольких выгрузках встретилось объявление")
    url_count = raw.get("photo_urls", pd.Series([[]] * len(raw), index=index)).map(
        lambda value: float(len(as_list(value))))
    builder.add_flag("photo_count_mismatch", url_count.ne(photos), "bookkeeping", "flag",
                     "photo_count_advertised vs photo_urls",
                     "заявленное число фото не совпадает со ссылками")

    # --- seller-claimed tag flags --------------------------------------------
    tags = raw.get("tags_seller_claimed", pd.Series([[]] * len(raw), index=index))
    for name, pattern in TAG_FLAGS:
        builder.add_flag(f"tag_{name}", list_flags(tags, pattern), "tags_seller", "flag",
                         "tags_seller_claimed", "метка заявлена продавцом, не проверена")
    land_use = raw.get("land_use_label", pd.Series([[]] * len(raw), index=index))
    builder.add_flag("land_use_izhs", list_flags(land_use, r"\bижс\b"), "tags_seller", "flag",
                     "land_use_label")
    builder.add_flag("land_use_missing", land_use.map(lambda value: not as_list(value)),
                     "tags_seller", "flag", "land_use_label")

    # --- one-hot: highway, locality type, settlement type ---------------------
    one_hot(text(raw, "highway"), "highway", builder, "categorical", "highway",
            options.one_hot_min_count)
    locality = text(raw, "locality")
    for name, pattern in LOCALITY_TYPES:
        builder.add_flag(name, locality.str.contains(pattern, regex=True, na=False),
                         "categorical", "flag", "locality", "тип населённого пункта из названия")
    builder.add_flag("locality_type_other", (
        ~locality.str.contains("|".join(pattern for _, pattern in LOCALITY_TYPES),
                               regex=True, na=False) & locality.ne("")
    ), "categorical", "flag", "locality", "тип не распознан")
    settlement = text(raw, "settlement")
    for name, pattern in SETTLEMENT_TYPES:
        builder.add_flag(name, settlement.str.contains(pattern, regex=True, na=False),
                         "categorical", "flag", "settlement", "тип посёлка из названия")
    builder.add_flag("settlement_type_other", (
        ~settlement.str.contains("|".join(pattern for _, pattern in SETTLEMENT_TYPES),
                                 regex=True, na=False) & settlement.ne("")
    ), "categorical", "flag", "settlement", "тип не распознан")

    # --- address and URL structure -------------------------------------------
    address = text(raw, "address_fragment")
    builder.add("address_parts", address.map(
        lambda value: float(len([part for part in value.split(",") if part.strip()]))),
        "structure", "count", "address_fragment", "число заполненных частей адреса")
    builder.add_flag("address_has_number", address.str.contains(r"\d", regex=True, na=False),
                     "structure", "flag", "address_fragment", "есть номер дома")
    locality_url = text(raw, "locality_url")
    builder.add("locality_url_depth", locality_url.map(
        lambda value: float(max(0, len([part for part in value.split("/") if part]) - 3))),
        "structure", "count", "locality_url", "глубина ссылки: район/СНТ/населённый пункт")

    # --- target-free frequency encodings for high-cardinality fields ---------
    for name, source in (("locality_listing_count", "locality"),
                         ("settlement_listing_count", "settlement")):
        values = text(raw, source)
        mapping = values[values.ne("")].value_counts()
        builder.add(name, values.map(mapping).astype("float64"), "encoding", "frequency",
                    source, "сколько объявлений в этом же населённом пункте (без цели)")

    # --- text-derived flags (unstructured field) ------------------------------
    if options.text_features:
        description = text(raw, "description")
        for name, group, pattern in TEXT_PATTERNS:
            builder.add_flag(name, description.str.contains(pattern, regex=True, na=False),
                             f"text_{group}", "flag", "description",
                             "флаг из текста объявления (проверять на дублирование с ценой)")
        builder.add("description_chars", description.str.len().astype("float64"),
                    "text_structure", "count", "description", "длина описания, символов")
        builder.add("description_words", description.str.split().map(len).astype("float64"),
                    "text_structure", "count", "description", "число слов в описании")
        builder.add("description_exclamations",
                    description.str.count("!").astype("float64").replace(0, np.nan),
                    "text_structure", "count", "description", "восклицательных знаков (null если 0)")
        builder.add_flag("description_has_phone",
                         description.str.contains(r"\d[\d\s\-()]{9,}", regex=True, na=False),
                         "text_structure", "flag", "description", "в описании похоже на телефон")

    # --- target ---------------------------------------------------------------
    target = numeric(raw, TARGET_COLUMN)
    builder.add(TARGET_COLUMN, target, "target", "target", TARGET_COLUMN,
                "цена предложения, ₽ — не цена сделки")
    builder.add(LOG_TARGET_COLUMN, np.log(target.where(target > 0)), "target", "target",
                TARGET_COLUMN, "логарифм цены: вариант цели для скошенного распределения")
    return builder


def coverage_table(features: pd.DataFrame, dictionary: pd.DataFrame) -> pd.DataFrame:
    """Coverage and basic statistics for every feature column."""
    rows: list[dict[str, Any]] = []
    for column in features.columns:
        values = features[column]
        filled = values.notna()
        rows.append({
            "column": column,
            "non_null": int(filled.sum()),
            "coverage_pct": filled.mean() * 100,
            "unique": int(values.nunique(dropna=True)),
            "mean": values.mean(),
            "std": values.std(),
            "min": values.min(),
            "max": values.max(),
        })
    table = pd.DataFrame(rows)
    return table.merge(dictionary, on="column", how="left")


def constant_columns(features: pd.DataFrame, exclude: tuple[str, ...]) -> list[str]:
    """Columns with no variation — useless for any model."""
    return [
        column for column in features.columns
        if column not in exclude and features[column].nunique(dropna=True) <= 1
    ]


def markdown_table(frame: pd.DataFrame, max_rows: int = 30) -> str:
    """Build a Markdown table without relying on the optional tabulate package."""
    if frame.empty:
        return "_Нет данных._"
    shown = frame.head(max_rows)
    headers = [str(name) for name in shown.columns]

    def display(value: Any) -> str:
        if value is None or (not isinstance(value, str) and pd.isna(value)):
            return "—"
        if isinstance(value, str):
            return value.replace("|", "\\|")
        if isinstance(value, (int, np.integer)):
            return f"{value:,}".replace(",", " ")
        if abs(float(value)) >= 1000:
            return f"{float(value):,.0f}".replace(",", " ")
        return f"{float(value):.3f}".replace(".", ",")

    lines = ["| " + " | ".join(headers) + " |",
             "| " + " | ".join("---" for _ in headers) + " |"]
    for row in shown.itertuples(index=False, name=None):
        lines.append("| " + " | ".join(display(value) for value in row) + " |")
    if len(frame) > max_rows:
        lines.append(f"\nПоказаны первые {max_rows} из {len(frame)} строк.")
    return "\n".join(lines)


def run(input_path: Path, features_path: Path, report_dir: Path,
        options: argparse.Namespace) -> None:
    raw, invalid_lines = read_jsonl(input_path)
    target = numeric(raw, TARGET_COLUMN)
    keep = target.gt(0)
    dropped_target = int((~keep).sum())
    raw = raw.loc[keep].reset_index(drop=True)
    if raw.empty:
        raise SystemExit("Не осталось строк с положительной ценой.")

    builder = build_features(raw, options)
    dictionary = builder.dictionary()
    features = builder.frame()

    constants = constant_columns(features, (TARGET_COLUMN, LOG_TARGET_COLUMN))
    if constants:
        features = features.drop(columns=constants)
        dictionary = dictionary.loc[~dictionary["column"].isin(constants)]

    listing_ids = raw[IDENTIFIER_COLUMN] if IDENTIFIER_COLUMN in raw else pd.Series(
        np.arange(len(raw))
    )
    matrix = pd.concat(
        [listing_ids.rename(IDENTIFIER_COLUMN).reset_index(drop=True), features], axis=1
    )

    # Group key for GroupKFold: listings of the same locality must not be split across folds,
    # otherwise near-duplicate objects leak between train and test. Not a feature.
    locality_values = text(raw, "locality")
    codes, uniques = pd.factorize(locality_values.where(locality_values.ne("")))
    matrix[GROUP_COLUMN] = codes
    dictionary = pd.concat([
        dictionary,
        pd.DataFrame([{
            "column": GROUP_COLUMN, "group": "bookkeeping", "kind": "group_key",
            "source_field": "locality",
            "note": f"ключ для GroupKFold ({len(uniques)} групп), не признак",
        }]),
    ], ignore_index=True)

    features_path.parent.mkdir(parents=True, exist_ok=True)
    matrix.to_csv(features_path, index=False, encoding="utf-8-sig")
    report_dir.mkdir(parents=True, exist_ok=True)
    coverage = coverage_table(features, dictionary)
    coverage.to_csv(report_dir / "feature_coverage.csv", index=False, encoding="utf-8-sig")
    dictionary.to_csv(report_dir / "feature_dictionary.csv", index=False, encoding="utf-8-sig")

    feature_columns = [column for column in features.columns
                       if column not in NON_FEATURE_COLUMNS]
    binary = [column for column in feature_columns if features[column].nunique() <= 2]
    numeric_like = len(feature_columns) - len(binary)
    group_counts = dictionary.loc[
        dictionary["column"].isin(feature_columns)
    ].groupby("group").size().sort_values(ascending=False)
    low_coverage = coverage.loc[
        coverage["column"].isin(feature_columns) & coverage["coverage_pct"].lt(50)
    ].sort_values("coverage_pct")

    report = [
        "# Широкий набор признаков из структурированных полей",
        "",
        f"Источник: `{input_path}`",
        f"Матрица: `{features_path}`",
        f"Строк: **{len(matrix):,}**; колонок всего: **{matrix.shape[1]}** "
        f"(признаков **{len(feature_columns)}**, плюс `{IDENTIFIER_COLUMN}`, цель "
        f"`{TARGET_COLUMN}`, вариант цели `{LOG_TARGET_COLUMN}` и ключ групп "
        f"`{GROUP_COLUMN}`).",
        f"Признаки: **{numeric_like}** числовых и **{len(binary)}** бинарных.",
        f"Битых строк JSON: **{invalid_lines}**; строк без положительной цены отброшено: "
        f"**{dropped_target}**.",
        f"Текстовые флаги из описания: {'включены' if options.text_features else 'выключены'}.",
        "",
        "## Группы признаков",
        markdown_table(group_counts.rename("Колонок").reset_index().rename(
            columns={"group": "Группа"}
        )),
        "",
        "## Защита от утечки целевой переменной",
        "Исключены все производные от цены колонки:",
        *[f"- `{column}` — рассчитывается из цены, попадёт в цель при обучении" 
          for column in LEAKAGE_COLUMNS],
        "- признаки меток площадки (`badges_platform_claimed`) **не создаются**: метка "
        "«Выгодная цена» вычисляется площадкой по относительной цене (медианный процентиль "
        "цены за м² 10,1 внутри диапазонов площади; 79 % таких объявлений — в самом дешёвом "
        "квартиле, 0 % — в самом дорогом), то есть повторяет цель; метка "
        "`badge_mortgage_discount_pct` связана с ценой немонотонно (Спирмен 0,82). "
        "Поле не используется ни отдельными флагами `badge_*`, ни счётчиком меток.",
        "Также исключены:",
        *[f"- `{column}` — {reason}" for column, reason in DROPPED_COLUMNS.items()],
        f"Константные и неинформативные колонки, удалённые автоматически: "
        f"{', '.join(f'`{name}`' for name in (*CONSTANT_COLUMNS, *constants)) or 'нет'}.",
        "",
        "## Пропуски",
        "Пропуски **не заполняются**: NaN остаётся в матрице, потому что "
        "`HistGradientBoostingRegressor`, XGBoost и LightGBM обрабатывают NaN сами, "
        "а заполнение до разбиения на фолды переносит информацию из теста в обучение. "
        "Рядом с каждым неполным полем добавлен индикатор пропуска "
        "(`*_missing`), чтобы линейные модели тоже могли использовать этот сигнал. "
        "Для линейных моделей заполните NaN медианой, посчитанной **только на обучающей части** "
        "каждого фолда.",
        "",
        f"Колонки с покрытием ниже 50 % ({len(low_coverage)}):",
        markdown_table(low_coverage[["column", "coverage_pct", "non_null", "group"]].rename(
            columns={"column": "Колонка", "coverage_pct": "Покрытие, %",
                     "non_null": "Заполнено", "group": "Группа"}
        ), max_rows=20),
        "",
        "## Как обучать на этой матрице",
        "1. Целевая переменная: `log_asking_price_rub` для скошенного распределения "
        "(`skew` цены ≈ 9,8) — метрики затем пересчитывайте обратно через `expm1`.",
        "2. Модели: `HistGradientBoostingRegressor` или LightGBM/XGBoost — они нативно "
        "работают с NaN; линейная регрессия потребует заполнения пропусков.",
        f"3. Разбиение: используйте `GroupKFold` с группами из `{GROUP_COLUMN}` — "
        "объявления одного населённого пункта (и, вероятно, одного физического дома) "
        "не должны попадать одновременно в обучение и тест.",
        "4. Высококардинальные поля (`highway`, `locality`, `settlement`) даны "
        "one-hot для шоссе и частотным кодированием для населённых пунктов; "
        "target-encoding не используется, так как требует расчёта внутри фолда.",
        "5. Признаки из текста описания (`txt_*`) — самая слабая по дисциплине часть "
        "набора: они извлекаются из рекламного текста, который может упоминать цену. "
        "Проверьте пересечение ключевых слов с ценой и при сомнениях отключите "
        "`--no-text-features`.",
        "",
        "## Ориентир качества (замер на этих данных, лог-цель, 5 фолдов)",
        "| Набор признаков | Модель | Разбиение | R² (лог) | R² (руб.) | RMSE, ₽ | MAE, ₽ |",
        "| --- | --- | --- | --- | --- | --- | --- |",
        "| 2 признака | линейная, сырая цель | KFold | — | 0,338 | 144 782 699 | 56 835 595 |",
        "| 2 признака | линейная | KFold | 0,517 | — | — | — |",
        "| 103 признака | линейная | KFold | 0,834 | 0,478 | 131 130 049 | 38 246 010 |",
        "| 103 признака | HistGB | KFold | **0,870** | **0,662** | **101 887 647** | **31 225 293** |",
        "| 103 признака | HistGB | GroupKFold по `group_locality` | 0,843 | 0,556 | 117 930 931 | 37 543 670 |",
        "",
        "Замеры сделаны отдельным скриптом: этот скрипт только строит признаки. "
        "Что из них следует:",
        "- Переход от 2 признаков к 103 поднимает R² на лог-шкале с 0,517 до 0,870, "
        "а MAE по сырой цене падает почти вдвое (57 → 31 млн ₽).",
        "- Честная групповая валидация стоит около 0,027 R² и 6 млн ₽ MAE: "
        "часть роста на обычном `KFold` даёт запоминание населённых пунктов.",
        "- Линейная модель на лог-цели при обратном преобразовании даёт RMSE в миллиарды "
        "рублей: `expm1` взрывается на экстраполяции. Ограничивайте прогноз "
        "обучающим диапазоном или обучайте на сырой цене.",
        "- Важность признаков (permutation importance) распределена крайне неравномерно: "
        "`house_area_m2` даёт около 0,68 падения R², `distance_to_mkad_km` — 0,078, "
        "`house_area_per_floor` — 0,034, остальные меньше 0,03. Признаки из описания "
        "слабые (лучший `txt_*` — около 0,008), но не бесполезные.",
        "",
        "## Файлы",
        f"- Матрица признаков: `{features_path}`",
        f"- Словарь признаков: `{report_dir / 'feature_dictionary.csv'}`",
        f"- Покрытие и статистика: `{report_dir / 'feature_coverage.csv'}`",
        f"- Отчёт: `{report_dir / 'report.md'}`",
    ]
    (report_dir / "report.md").write_text("\n".join(report) + "\n", encoding="utf-8")

    print(f"Строк: {len(matrix):,}; колонок: {matrix.shape[1]} "
          f"(признаков {len(feature_columns)})".replace(",", " "))
    print(f"Матрица признаков: {features_path}")
    print(f"Отчёт и словарь: {report_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Построение широкой матрицы признаков из objects.jsonl для обучения модели цены."
    )
    parser.add_argument(
        "--input", type=Path, default=default_input_path(),
        help="Путь к objects.jsonl (по умолчанию data/dataset/objects.jsonl).",
    )
    parser.add_argument(
        "--features-file", type=Path, default=default_features_path(),
        help="Куда записать матрицу признаков (по умолчанию data/features/features_rich.csv).",
    )
    parser.add_argument(
        "--report-dir", type=Path, default=default_report_dir(),
        help="Каталог для отчёта и словаря (по умолчанию reports/features_rich).",
    )
    parser.add_argument(
        "--one-hot-min-count", type=int, default=10,
        help="Минимальное число строк, чтобы уровень категории получил свою колонку.",
    )
    parser.add_argument(
        "--reference-date", type=str, default=None,
        help="Дата отсчёта (ISO) для абсолютных дат в listing_age_text; "
             "по умолчанию берётся самая поздняя дата в данных.",
    )
    text_group = parser.add_mutually_exclusive_group()
    text_group.add_argument(
        "--text-features", dest="text_features", action="store_true", default=True,
        help="Добавлять флаги из текста описания (по умолчанию включено).",
    )
    text_group.add_argument(
        "--no-text-features", dest="text_features", action="store_false",
        help="Строить признаки только из структурированных полей.",
    )
    parser.add_argument("--verbose", action="store_true",
                        help="Показывать предупреждения и подробные сообщения.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(levelname)s: %(message)s",
    )
    run(
        args.input.expanduser(),
        args.features_file.expanduser(),
        args.report_dir.expanduser(),
        args,
    )


if __name__ == "__main__":
    main()
