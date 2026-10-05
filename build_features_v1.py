#!/usr/bin/env python3
"""Build the compact feature matrix for house-price models: numeric backbone + 10 groups.

Source:  data/dataset/objects.jsonl — structured fields of the Domclick export.
         data/features/llm_extractions.jsonl — answers of the LLM extraction
         (`build_llm_extractions.py`, base prompt `prompts/prompt_v1.txt`,
         `prompt_version = 1.0`); they fill the gaps and build the groups.
Output:  data/features/features_v1.csv — one row per listing, numeric only.
         reports/features_v1/ — feature dictionary, coverage, report.

Числовой каркас — из карточки объявления, пропуски заполняются извлечениями:
    house_area_m2, distance_to_mkad_km, highway, land_area_sotka, floors,
    house_area_per_floor, land_to_house_ratio

Группы — собираются из извлечений, у каждой свой канонический словарь уровней:
    Земля ................ ИЖС / СНТ / ДНП / ЛПХ / земли населённых пунктов / сельхоз / другие
    Отделка .............. без отделки / предчистовая / готовая / частичная / неизвестно
    Газ .................. подключён / по границе / рядом / отсутствует / неизвестно
    Стены ................ газобетон / пенобетон / кирпич / каркас / брус / бревно / другие
    Фундамент ............ плита / ленточный / сваи / свайно-ростверковый / другой
    Отопление ............ источник тепла и заявленная работоспособность
    Вода и канализация ... тип системы и состояние подключения
    Планировка ........... спальни, санузлы, мансарда, подвал
    Дополнительные строения ... баня, гараж, гостевой дом и их площади
    Возраст .............. год постройки и реконструкции

Design rules
------------
1. The structured export is the primary source for the numeric backbone and for the land
   category (its card tags); the extraction only fills their gaps.
2. A group value is taken only when its status is in `--llm-statuses` (default
   `stated`): the prompt distinguishes existing, planned and possible facts, and
   "газ по границе" is not connected gas.
3. Free-text levels are collapsed into the canonical vocabulary of their group
   ("газобетонные блоки", "газоблок", "газобетон" -> `gazobeton`); everything
   unrecognised becomes `other`, absent values become `unknown`. The vocabulary
   columns are always emitted, so the schema does not depend on the sample.
4. Missing values stay NaN instead of being imputed: HistGradientBoosting, XGBoost
   and LightGBM handle NaN natively, and imputing before a train/test split leaks
   information.
5. Nothing derived from the price is built: price columns would leak the target, and
   the platform badge "Выгодная цена" is computed from the price as well.
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


LOG = logging.getLogger("domclick.features_v1")

IDENTIFIER_COLUMN = "listing_id"
TARGET_COLUMN = "asking_price_rub"
LOG_TARGET_COLUMN = "log_asking_price_rub"
GROUP_COLUMN = "group_locality"

# Carried for training and grouped validation, never used as features.
NON_FEATURE_COLUMNS = (IDENTIFIER_COLUMN, TARGET_COLUMN, LOG_TARGET_COLUMN, GROUP_COLUMN)

BACKBONE_GROUP = "числовой каркас"

# Thematic groups of the matrix, in the order they appear in the report.
FEATURE_GROUPS: tuple[str, ...] = (
    BACKBONE_GROUP,
    "Земля",
    "Отделка",
    "Газ",
    "Стены",
    "Фундамент",
    "Отопление",
    "Вода и канализация",
    "Планировка",
    "Дополнительные строения",
    "Возраст",
)

# (kind, source, note) of the numeric backbone; the order of the columns is fixed below.
BACKBONE_META: dict[str, tuple[str, str, str]] = {
    "house_area_m2": (
        "numeric", "objects.jsonl:house_area_m2 | llm:house_area_m2",
        "площадь дома, м²; пропуски карточки заполнены значением из llm_extractions.jsonl",
    ),
    "distance_to_mkad_km": (
        "numeric", "objects.jsonl:distance_to_mkad_km | llm:distance_to_mkad_km",
        "расстояние до МКАД, км; пропуски карточки заполнены из llm_extractions.jsonl",
    ),
    "highway": (
        "categorical", "objects.jsonl:highway | llm:highway",
        "шоссе; пустые значения заполнены из llm_extractions.jsonl, затем one-hot",
    ),
    "land_area_sotka": (
        "numeric", "objects.jsonl:land_area_sotka | llm:land_area_sotka",
        "площадь участка, соток; пропуски карточки заполнены из llm_extractions.jsonl",
    ),
    "floors": (
        "numeric", "objects.jsonl:floors | llm:floors",
        "этажность; пропуски карточки заполнены из llm_extractions.jsonl",
    ),
    "house_area_per_floor": (
        "derived", "house_area_m2 / floors",
        "средняя площадь этажа, м²; считается после заполнения пропусков",
    ),
    "land_to_house_ratio": (
        "derived", "land_area_sotka / house_area_m2",
        "баланс участка и дома; считается после заполнения пропусков",
    ),
}

BACKBONE_ORDER: tuple[str, ...] = (
    "house_area_m2",
    "distance_to_mkad_km",
    "highway",
    "land_area_sotka",
    "floors",
    "house_area_per_floor",
    "land_to_house_ratio",
)

# Backbone families that objects.jsonl provides; the extractions only fill their gaps.
STRUCTURED_FEATURES = ("house_area_m2", "distance_to_mkad_km", "highway",
                       "land_area_sotka", "floors")

# Canonical categorical groups. `levels` are ordered rules (canonical level, regex);
# the first matching rule wins. `other` adds a catch-all level for unrecognised text;
# enum groups (Отделка, Газ) have no catch-all, an unknown vocabulary value stays empty.
CATEGORICAL_GROUPS: tuple[dict[str, Any], ...] = (
    {
        "group": "Земля",
        "prefix": "land_category",
        "card_field": "land_use_label",
        "llm_fields": ("permitted_use", "land_category"),
        "levels": (("izhs", r"\bижс\b|индивидуальн\w* жил|индивидуальное жилищное|"
                            r"жилой застройк|жилыми зданиями"),
                   ("dnp", r"\bднп\b|дачн"),
                   ("snt", r"\bснт\b|садовод|огороднич|коллективн\w* хозяйств"),
                   ("lph", r"\bлпх\b|личн\w* подсобн|подсобн\w* хозяйств"),
                   ("znp", r"\bзнп\b|земл\w* насел[её]нн|насел[её]нных пунктов|"
                           r"земл\w* поселен"),
                   ("selkhoz", r"сельскохозяйственн|сельхозназнач|земл\w* с/х|с/х")),
        "other": True,
        "note": "категория земель и вид разрешённого использования: ИЖС / СНТ / ДНП / ЛПХ / "
                "земли населённых пунктов / сельхозназначение / другие; источник — метки "
                "карточки, затем вид разрешённого использования, затем категория земель",
    },
    {
        "group": "Отделка",
        "prefix": "finishing_stage",
        "field": "finishing_stage",
        "levels": (("bez_otdelki", r"^none$"), ("predchistovaya", r"^pre_finish$"),
                   ("gotovaya", r"^finished$"), ("chastichnaya", r"^partial$")),
        "other": False,
        "note": "отделка: без отделки / предчистовая / готовая / частичная",
    },
    {
        "group": "Газ",
        "prefix": "gas_status",
        "field": "gas_status",
        "levels": (("podklyuchen", r"^connected$"), ("po_granice", r"^at_boundary$"),
                   ("ryadom", r"^nearby$"), ("otsutstvuet", r"^absent$")),
        "other": False,
        "note": "газ: подключён / по границе / рядом / отсутствует",
    },
    {
        "group": "Стены",
        "prefix": "wall_material",
        "field": "wall_material",
        "levels": (("gazobeton", r"газобетон|газоблок|газосиликат"),
                   ("penobeton", r"пенобетон|пеноблок"),
                   ("kirpich", r"кирпич"),
                   ("karkas", r"каркас|сип|щитов"),
                   ("brus", r"брус"),
                   ("brevno", r"бревн|бревен|оцилиндрован|сруб")),
        "other": True,
        "note": "стены: газобетон / пенобетон / кирпич / каркас / брус / бревно / другие; "
                "«дерево» и блоки без уточнения попадают в «другие»",
    },
    {
        "group": "Фундамент",
        "prefix": "foundation_type",
        "field": "foundation_type",
        "levels": (("svayno_rostverkovyy", r"ростверк"),
                   ("lentochny", r"ленточн|лента|фбс"),
                   ("plita", r"плит|ушп|монолит"),
                   ("svai", r"сва[йия]|буронабивн|винтов")),
        "other": True,
        "note": "фундамент: плита / ленточный / сваи / свайно-ростверковый / другой",
    },
    {
        "group": "Отопление",
        "prefix": "heating_type",
        "field": "heating_type",
        "levels": (("gazovoe", r"газ"),
                   ("elektricheskoe", r"электр|конвектор|обогрев|инфракрасн|теплов[оа]й насос"),
                   ("pechnoe", r"печ|камин"),
                   ("tverdotoplivnoe", r"твердотопливн|дров|уголь|пеллет")),
        "other": True,
        "note": "отопление: газовое / электрическое / печное / твёрдотопливное / другое; "
                "«тёплые полы» и «центральное» без источника — в «другое»",
    },
    {
        "group": "Вода и канализация",
        "prefix": "water_source",
        "field": "water_source",
        "levels": (("skvazhina", r"скважин|артезиан"),
                   ("kolodec", r"колодец|колодц"),
                   ("other", r"летн"),
                   ("centralnoe", r"центральн|централизован|водопровод|городск|магистраль"),
                   ("privoza", r"привоз|завоз|доставк")),
        "other": True,
        "note": "водоснабжение: скважина / колодец / центральное / привозная вода / другое",
    },
    {
        "group": "Вода и канализация",
        "prefix": "sewage_type",
        "field": "sewage_type",
        "levels": (("centralnoe", r"центральн|централизован|поселков"),
                   ("septik", r"септик|септ|био|топас|юнилос|автономн|локальн|очистк"),
                   ("vygrebnaya_yama", r"выгребн|яма")),
        "other": True,
        "note": "канализация: центральное / септик (включая станции биоочистки) / "
                "выгребная яма / другое",
    },
)

# (group, column, extraction field, kind, note); kind ∈ {count, flag, year}.
NUMBER_FEATURES: tuple[tuple[str, str, str, str, str], ...] = (
    ("Планировка", "bedrooms", "bedrooms", "count", "число спален"),
    ("Планировка", "bathrooms", "bathrooms", "count", "число санузлов"),
    ("Планировка", "attic_present", "attic_present", "flag", "заявлена мансарда"),
    ("Планировка", "basement_present", "basement_present", "flag",
     "заявлен подвал или цокольный этаж"),
    ("Отопление", "heating_operational", "heating_operational", "flag",
     "заявлена работоспособность отопления"),
    ("Вода и канализация", "water_connected", "water_connected", "flag",
     "водоснабжение подключено"),
    ("Вода и канализация", "sewage_connected", "sewage_connected", "flag",
     "канализация подключена"),
    ("Возраст", "construction_year", "construction_year", "year", "год постройки"),
    ("Возраст", "renovation_year", "renovation_year", "year",
     "год реконструкции, если она была"),
)

# Additional buildings come from a top-level list of the extraction, not from `features`.
BUILDING_GROUP = "Дополнительные строения"
BUILDING_SPECS: tuple[tuple[str, str, str], ...] = (
    ("bathhouse", r"бан|саун", "баня или сауна"),
    ("garage", r"гараж", "гараж"),
    ("guest_house", r"гостевой дом", "гостевой дом"),
)

# Column order of the group part of the matrix: families, not data types.
EMIT_ORDER: tuple[str, ...] = (
    "land_category",
    "finishing_stage",
    "gas_status",
    "wall_material",
    "foundation_type",
    "heating_type", "heating_operational",
    "water_source", "water_connected", "sewage_type", "sewage_connected",
    "bedrooms", "bathrooms", "attic_present", "basement_present",
    "additional_buildings_count", "bathhouse_present", "bathhouse_area_m2",
    "garage_present", "garage_area_m2", "guest_house_present", "guest_house_area_m2",
    "construction_year", "renovation_year",
)

# Level that collects values outside the canonical vocabulary of a free-text group.
OTHER_LEVEL = "other"
UNKNOWN_LEVEL = "unknown"

# Plausible range for the year fields: everything else is a data error, not a year.
YEAR_RANGE = (1900.0, 2100.0)

# Tokens that mean True in a boolean extraction value (mirrors coerce_value).
TRUE_TOKENS = frozenset({"true", "да", "yes", "есть", "1"})

# The prompt distinguishes existing, planned and possible facts; only existing ones
# describe the object as it is now, so only these statuses may fill a value.
DEFAULT_LLM_STATUSES = ("stated",)

# Base prompt of the extraction; it is written into every cached answer.
PROMPT_VERSION = "1.0"

# Values that mean "the field is empty" rather than a level of the characteristic.
EMPTY_TEXT_VALUES = frozenset({"", "-", "нет", "нет данных", "не указано"})


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


def default_llm_path() -> Path:
    """Cache of the LLM answers, written by build_llm_extractions.py."""
    return Path(__file__).resolve().parent / "data" / "features" / "llm_extractions.jsonl"


def default_features_path() -> Path:
    """The compact matrix belongs in data/, next to the other feature matrices."""
    return Path(__file__).resolve().parent / "data" / "features" / "features_v1.csv"


def default_report_dir() -> Path:
    """Reports belong in reports/, separate from the data."""
    return Path(__file__).resolve().parent / "reports" / "features_v1"


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


def clean_categorical(values: pd.Series) -> pd.Series:
    """Normalized categorical text: lowercase, collapsed spaces, no trailing punctuation."""
    cleaned = values.astype("string").fillna("").str.strip().str.lower()
    cleaned = cleaned.str.replace("\u00a0", " ", regex=False)
    cleaned = cleaned.str.replace(r"\s+", " ", regex=True)
    cleaned = cleaned.str.replace(r"[.,;]+$", "", regex=True)
    return cleaned.mask(cleaned.isin(EMPTY_TEXT_VALUES))


def land_use_tags(raw: pd.DataFrame, index: pd.Index) -> pd.Series:
    """Seller-claimed land-use tags of the card as one normalized string per listing."""
    if "land_use_label" not in raw:
        return pd.Series("", index=index, dtype="string")
    return raw["land_use_label"].map(
        lambda items: "; ".join(str(item).strip() for item in items if str(item).strip())
        if isinstance(items, list) else ""
    )


def positive(values: pd.Series) -> pd.Series:
    """Keep only positive numbers; zero and negative values are not characteristics."""
    return values.where(values > 0)


def to_numeric(values: pd.Series) -> pd.Series:
    """Numeric view of raw extraction values: non-numeric, infinite and negative -> NaN."""
    numbers = pd.to_numeric(values, errors="coerce").astype("float64")
    numbers = numbers.replace([np.inf, -np.inf], np.nan)
    return numbers.where(numbers >= 0)


def to_flag(values: pd.Series) -> pd.Series:
    """Boolean extraction value as 0/1; a missing value stays NaN."""
    def one(value: Any) -> float:
        if value is None:
            return np.nan
        if isinstance(value, bool):
            return float(value)
        if isinstance(value, (int, float)):
            return float(bool(value))
        token = str(value).strip().lower()
        if token in TRUE_TOKENS:
            return 1.0
        return 0.0 if token else np.nan

    return values.map(one).astype("float64")


def to_year(values: pd.Series) -> pd.Series:
    """Year within a plausible range; anything else is treated as missing."""
    years = to_numeric(values)
    return years.where(years.between(*YEAR_RANGE))


def vocabulary_of(spec: dict[str, Any]) -> list[str]:
    """Canonical levels of a group (without `unknown`), in their documented order."""
    level_names = [level for level, _ in spec["levels"]]
    if spec["other"]:
        level_names.append(OTHER_LEVEL)
    return list(dict.fromkeys(level_names))


def canonical_levels(values: pd.Series, rules: tuple[tuple[str, str], ...],
                     other: bool) -> pd.Series:
    """Map raw values to the canonical levels of a group; NA means "unknown".

    Rules are ordered: the first matching regex wins, so a narrower rule must come
    before a broader one. Unrecognised text becomes `other` when the group has a
    catch-all level.
    """
    cleaned = clean_categorical(values)
    # pandas 3 хранит текст в pyarrow-колонке, а её векторный regex — это RE2, где \b и
    # \w понимают только ASCII: шаблон вроде \bижс\b молча не находит кириллицу. Поэтому
    # сопоставление идёт по object-колонке, где pandas использует модуль re.
    matcher = cleaned.astype(object)
    levels = pd.Series(pd.NA, index=values.index, dtype="string")
    for level, pattern in rules:
        matched = matcher.str.contains(pattern, regex=True, na=False) & levels.isna()
        levels = levels.mask(matched, level)
    if other:
        levels = levels.mask(cleaned.notna() & levels.isna(), OTHER_LEVEL)
    return levels


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
        note="пустое значение",
    )
    created.append(f"{prefix}_unknown")
    return created


def load_extractions(path: Path) -> tuple[dict[str, dict[str, Any]], dict[str, int]]:
    """Read the LLM cache: `listing_id` -> cached record (the last record wins)."""
    records: dict[str, dict[str, Any]] = {}
    stats = {"records": 0, "invalid_lines": 0, "duplicates": 0}
    try:
        content = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise SystemExit(f"Не удалось прочитать {path}: {exc}") from exc
    for line in content.splitlines():
        if not line.strip():
            continue
        try:
            item = json.loads(line)
        except json.JSONDecodeError:
            stats["invalid_lines"] += 1
            continue
        if not isinstance(item, dict) or item.get("listing_id") is None:
            stats["invalid_lines"] += 1
            continue
        listing_id = str(item["listing_id"])
        if listing_id in records:
            stats["duplicates"] += 1
        records[listing_id] = item
        stats["records"] += 1
    return records, stats


def extraction_features(records: dict[str, dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """`features` block of every cached answer, keyed by listing id."""
    features: dict[str, dict[str, Any]] = {}
    for listing_id, record in records.items():
        extraction = record.get("extraction")
        block = extraction.get("features") if isinstance(extraction, dict) else None
        features[listing_id] = block if isinstance(block, dict) else {}
    return features


def extraction_buildings(records: dict[str, dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    """`additional_buildings` list of every cached answer, keyed by listing id."""
    buildings: dict[str, list[dict[str, Any]]] = {}
    for listing_id, record in records.items():
        extraction = record.get("extraction")
        items = extraction.get("additional_buildings") if isinstance(extraction, dict) else None
        buildings[listing_id] = [item for item in items if isinstance(item, dict)] \
            if isinstance(items, list) else []
    return buildings


def extraction_values(features_by_id: dict[str, dict[str, Any]], ids: pd.Series,
                      field: str, statuses: tuple[str, ...]) -> pd.Series:
    """Trusted value of one extraction field per listing.

    None whenever the answer is missing, the value is null or the status is not in
    `statuses` — a planned or possible fact is not a characteristic of the object.
    """
    values: list[Any] = []
    for listing_id in ids:
        entry = (features_by_id.get(str(listing_id)) or {}).get(field)
        if not isinstance(entry, dict) or entry.get("status") not in statuses:
            values.append(None)
            continue
        values.append(entry.get("value"))
    return pd.Series(values, index=ids.index, dtype="object")


def fill_gaps(name: str, group: str, structured: pd.Series, fallback: pd.Series,
              sources: dict[str, dict[str, Any]]) -> pd.Series:
    """Use the card value where it exists and the extraction value where it does not."""
    missing = structured.isna()
    combined = structured.where(~missing, fallback)
    sources[name] = {
        "group": group,
        "kind": "карточка + LLM",
        "structured": int((~missing).sum()),
        "from_llm": int((missing & fallback.notna()).sum()),
        "missing": int(combined.isna().sum()),
    }
    return combined


def count_disagreements(structured: pd.Series, fallback: pd.Series, kind: str) -> int:
    """Rows where both sources have a value but the values differ."""
    both = structured.notna() & fallback.notna()
    if not both.any():
        return 0
    if kind == "numeric":
        left = structured[both].astype("float64")
        right = fallback[both].astype("float64")
        scale = pd.concat([left.abs(), right.abs()], axis=1).max(axis=1).replace(0, np.nan)
        relative = (left - right).abs() / scale
        return int(relative.gt(0.01).fillna(False).sum())
    return int((structured[both].astype("string") != fallback[both].astype("string")).sum())


def status_summary(features_by_id: dict[str, dict[str, Any]], fields: tuple[str, ...],
                   statuses: tuple[str, ...]) -> pd.DataFrame:
    """Per field: how many extracted values are used and how many are dropped by status."""
    rows: list[dict[str, Any]] = []
    for field in fields:
        used = dropped = 0
        for features in features_by_id.values():
            entry = features.get(field)
            if not isinstance(entry, dict) or entry.get("value") is None:
                continue
            if entry.get("status") in statuses:
                used += 1
            else:
                dropped += 1
        rows.append({"feature": field, "принято": used, "отброшено по статусу": dropped})
    return pd.DataFrame(rows)


def building_series(buildings_by_id: dict[str, list[dict[str, Any]]], ids: pd.Series,
                    answered: pd.Series, pattern: str) -> tuple[pd.Series, pd.Series]:
    """Presence and the largest area of one building type, per listing."""
    regex = re.compile(pattern)
    present: list[float] = []
    areas: list[float] = []
    for listing_id, known in zip(ids, answered):
        if not known:
            present.append(np.nan)
            areas.append(np.nan)
            continue
        items = buildings_by_id.get(str(listing_id)) or []
        matched = [item for item in items if regex.search(str(item.get("type") or "").lower())]
        present.append(1.0 if matched else 0.0)
        sizes = [float(item["area_m2"]) for item in matched
                 if isinstance(item.get("area_m2"), (int, float)) and item["area_m2"] > 0]
        areas.append(max(sizes) if sizes else np.nan)
    return (pd.Series(present, index=ids.index, dtype="float64"),
            pd.Series(areas, index=ids.index, dtype="float64"))


def build_features(raw: pd.DataFrame, features_by_id: dict[str, dict[str, Any]],
                   buildings_by_id: dict[str, list[dict[str, Any]]],
                   options: argparse.Namespace) -> tuple[FeatureBuilder, dict[str, Any]]:
    """Build the matrix: the numeric backbone plus the canonical groups."""
    builder = FeatureBuilder()
    index = raw.index
    if IDENTIFIER_COLUMN in raw:
        ids = raw[IDENTIFIER_COLUMN]
    else:
        ids = pd.Series(np.arange(len(raw)), index=index)
    answered = pd.Series([str(listing_id) in features_by_id for listing_id in ids],
                         index=index)
    statuses = tuple(options.llm_statuses)

    sources: dict[str, dict[str, Any]] = {}

    def llm_number(name: str, kind: str) -> pd.Series:
        values = extraction_values(features_by_id, ids, name, statuses)
        return to_year(values) if kind == "year" else to_numeric(values)

    # --- числовой каркас: карточка первична, извлечения заполняют пропуски ---------
    structured = {
        "house_area_m2": positive(numeric(raw, "house_area_m2")),
        "land_area_sotka": positive(numeric(raw, "land_area_sotka")),
        "floors": positive(numeric(raw, "floors")),
        "distance_to_mkad_km": positive(numeric(raw, "distance_to_mkad_km")),
        "highway": clean_categorical(raw.get(
            "highway", pd.Series("", index=index, dtype="string"))),
    }
    fallback = {
        "house_area_m2": positive(to_numeric(
            extraction_values(features_by_id, ids, "house_area_m2", statuses))),
        "land_area_sotka": positive(to_numeric(
            extraction_values(features_by_id, ids, "land_area_sotka", statuses))),
        "floors": positive(to_numeric(
            extraction_values(features_by_id, ids, "floors", statuses))),
        "distance_to_mkad_km": positive(to_numeric(
            extraction_values(features_by_id, ids, "distance_to_mkad_km", statuses))),
        "highway": clean_categorical(
            extraction_values(features_by_id, ids, "highway", statuses)),
    }
    backbone: dict[str, pd.Series] = {}
    for name in STRUCTURED_FEATURES:
        backbone[name] = fill_gaps(name, BACKBONE_GROUP, structured[name], fallback[name],
                                   sources)
    for name in ("house_area_per_floor", "land_to_house_ratio"):
        backbone[name] = (
            backbone["house_area_m2"] / backbone["floors"].where(backbone["floors"] > 0)
            if name == "house_area_per_floor"
            else backbone["land_area_sotka"]
            / backbone["house_area_m2"].where(backbone["house_area_m2"] > 0)
        )
        sources[name] = {"group": BACKBONE_GROUP, "kind": "производный", "structured": 0,
                         "from_llm": 0, "missing": int(backbone[name].isna().sum())}
    disagreements = {
        name: count_disagreements(structured[name], fallback[name],
                                  "text" if name == "highway" else "numeric")
        for name in STRUCTURED_FEATURES
    }

    # --- канонические категориальные группы ---------------------------------------
    levels_by_prefix: dict[str, pd.Series] = {}
    specs_by_prefix: dict[str, dict[str, Any]] = {}
    level_rows: list[dict[str, Any]] = []
    unmapped: dict[str, int] = {}
    for spec in CATEGORICAL_GROUPS:
        prefix, group = spec["prefix"], spec["group"]
        if spec.get("card_field"):
            # Метки карточки приоритетны, дальше поля извлечений в указанном порядке.
            card_raw = land_use_tags(raw, index)
            card = canonical_levels(card_raw, spec["levels"], bool(spec["other"]))
            unrecognised = int((clean_categorical(card_raw).notna() & card.isna()).sum())
            fallback = pd.Series(pd.NA, index=index, dtype="string")
            for field in spec["llm_fields"]:
                raw_values = extraction_values(features_by_id, ids, field, statuses)
                values = canonical_levels(raw_values, spec["levels"], bool(spec["other"]))
                unrecognised += int(
                    (clean_categorical(raw_values).notna() & values.isna()).sum())
                fallback = fallback.where(fallback.notna(), values)
            levels = fill_gaps(prefix, group, card, fallback, sources)
            source = (f"objects.jsonl:{spec['card_field']} | "
                      + " | ".join(f"llm:{field}" for field in spec["llm_fields"]))
            unmapped[prefix] = unrecognised
        else:
            raw_values = extraction_values(features_by_id, ids, spec["field"], statuses)
            levels = canonical_levels(raw_values, spec["levels"], bool(spec["other"]))
            source = f"llm:{spec['field']}"
            unmapped[prefix] = int(
                (clean_categorical(raw_values).notna() & levels.isna()).sum())
            sources[prefix] = {
                "group": group,
                "kind": "только LLM",
                "structured": 0,
                "from_llm": int(levels.notna().sum()),
                "missing": int(levels.isna().sum()),
            }
        levels_by_prefix[prefix] = levels
        specs_by_prefix[prefix] = {**spec, "source": source}
        for level in (*vocabulary_of(spec), UNKNOWN_LEVEL):
            mask = levels.isna() if level == UNKNOWN_LEVEL else levels.eq(level)
            level_rows.append({
                "группа": group, "признак": prefix, "уровень": level,
                "строк": int(mask.fillna(False).sum()),
            })

    # --- числовые и флаговые признаки групп ---------------------------------------
    number_series: dict[str, pd.Series] = {}
    for group, column, field, kind, note in NUMBER_FEATURES:
        values = (to_flag(extraction_values(features_by_id, ids, field, statuses))
                  if kind == "flag" else llm_number(field, kind))
        # Значение есть только там, где модель ответила по этому объявлению.
        number_series[column] = values.where(answered)
        sources[column] = {
            "group": group,
            "kind": "только LLM",
            "structured": 0,
            "from_llm": int(number_series[column].notna().sum()),
            "missing": int(number_series[column].isna().sum()),
        }

    # --- дополнительные строения ---------------------------------------------------
    building_series_map: dict[str, pd.Series] = {}
    counts = pd.Series(
        [float(len(buildings_by_id.get(str(listing_id)) or [])) for listing_id in ids],
        index=index, dtype="float64",
    ).where(answered)
    building_series_map["additional_buildings_count"] = counts
    sources["additional_buildings_count"] = {
        "group": BUILDING_GROUP, "kind": "только LLM", "structured": 0,
        "from_llm": int(counts.notna().sum()), "missing": int(counts.isna().sum()),
    }
    for name, pattern, _note in BUILDING_SPECS:
        present, area = building_series(buildings_by_id, ids, answered, pattern)
        building_series_map[f"{name}_present"] = present
        building_series_map[f"{name}_area_m2"] = area
        for column, values in ((f"{name}_present", present), (f"{name}_area_m2", area)):
            sources[column] = {
                "group": BUILDING_GROUP, "kind": "только LLM", "structured": 0,
                "from_llm": int(values.notna().sum()),
                "missing": int(values.isna().sum()),
            }

    # --- вывод колонок в фиксированном порядке ------------------------------------
    for name in BACKBONE_ORDER:
        kind, source, note = BACKBONE_META[name]
        if name == "highway":
            one_hot(backbone[name], name, builder, BACKBONE_GROUP, source,
                    options.one_hot_min_count)
        else:
            builder.add(name, backbone[name], BACKBONE_GROUP, kind, source, note)

    group_columns: dict[str, pd.Series] = {
        **number_series, **building_series_map,
    }
    number_meta = {column: (group, kind, note)
                   for group, column, _field, kind, note in NUMBER_FEATURES}
    building_notes = {
        "additional_buildings_count": "число дополнительных строений в извлечении",
        **{f"{name}_present": f"в извлечении есть {note} ({pattern})"
           for name, pattern, note in BUILDING_SPECS},
        **{f"{name}_area_m2": f"наибольшая площадь: {note}"
           for name, pattern, note in BUILDING_SPECS},
    }
    for name in EMIT_ORDER:
        if name in specs_by_prefix:
            spec = specs_by_prefix[name]
            levels = levels_by_prefix[name]
            for level in vocabulary_of(spec):
                builder.add_flag(
                    f"{name}_{level}", levels.eq(level), spec["group"], "level",
                    spec["source"], note=f"{spec['note']}; уровень: {level}",
                )
            builder.add_flag(
                f"{name}_{UNKNOWN_LEVEL}", levels.isna(), spec["group"], "level",
                spec["source"], note=f"{spec['note']}; уровень: {UNKNOWN_LEVEL}",
            )
            continue
        if name in group_columns:
            group, kind, note = number_meta.get(
                name, (BUILDING_GROUP, "count", building_notes.get(name, "")))
            builder.add(name, group_columns[name], group, kind, f"llm:{name}", note)
            continue
        raise AssertionError(f"Признак {name} не описан.")

    target = numeric(raw, TARGET_COLUMN)
    builder.add(TARGET_COLUMN, target, "target", "target", TARGET_COLUMN,
                "цена предложения, ₽ — не цена сделки")
    builder.add(LOG_TARGET_COLUMN, np.log(target.where(target > 0)), "target", "target",
                TARGET_COLUMN, "логарифм цены: вариант цели для скошенного распределения")

    audit = {
        "sources": sources,
        "disagreements": disagreements,
        "statuses": status_summary(
            features_by_id,
            tuple(dict.fromkeys(
                [field for spec in CATEGORICAL_GROUPS
                 for field in ([spec["field"]] if spec.get("field") else spec["llm_fields"])]
                + [field for _g, _c, field, _k, _n in NUMBER_FEATURES])),
            statuses),
        "levels": pd.DataFrame(level_rows),
        "unmapped": unmapped,
    }
    return builder, audit


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


def counted(value: int) -> str:
    """Group thousands with a space, the way the other reports do."""
    return f"{value:,}".replace(",", " ")


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
            return counted(int(value))
        number = float(value)
        if number.is_integer():
            return counted(int(number))
        return f"{number:.1f}".replace(".", ",")

    lines = ["| " + " | ".join(headers) + " |",
             "| " + " | ".join("---" for _ in headers) + " |"]
    for row in shown.itertuples(index=False, name=None):
        lines.append("| " + " | ".join(display(value) for value in row) + " |")
    if len(frame) > max_rows:
        lines.append(f"\nПоказаны первые {max_rows} из {len(frame)} строк.")
    return "\n".join(lines)


def family_table(audit: dict[str, Any], row_count: int) -> pd.DataFrame:
    """Per feature family: its group, sources of values and the resulting coverage."""
    rows: list[dict[str, Any]] = []
    for name in BACKBONE_ORDER + EMIT_ORDER:
        stats = audit["sources"][name]
        filled = row_count - stats["missing"]
        rows.append({
            "группа": stats["group"],
            "признак": name,
            "источник": stats["kind"],
            "из карточки": stats["structured"],
            "из LLM": stats["from_llm"],
            "пропусков": stats["missing"],
            "покрытие, %": round(filled / row_count * 100, 1) if row_count else 0.0,
        })
    return pd.DataFrame(rows)


def group_table(dictionary: pd.DataFrame, audit: dict[str, Any],
                row_count: int) -> pd.DataFrame:
    """Per group: how many columns it produced and the mean coverage of its families."""
    feature_kinds = {"numeric", "level", "flag", "count", "year", "one_hot", "derived"}
    columns = dictionary.loc[dictionary["kind"].isin(feature_kinds)]
    columns_per_group = columns.groupby("group").size()
    coverage: dict[str, list[float]] = {}
    for stats in audit["sources"].values():
        filled = row_count - stats["missing"]
        coverage.setdefault(stats["group"], []).append(
            filled / row_count * 100 if row_count else 0.0)
    rows: list[dict[str, Any]] = []
    for group in FEATURE_GROUPS:
        count = int(columns_per_group.get(group, 0))
        if not count:
            continue
        values = coverage.get(group) or [0.0]
        rows.append({
            "группа": group,
            "колонок": count,
            "семейств": len(values),
            "среднее покрытие, %": round(sum(values) / len(values), 1),
        })
    return pd.DataFrame(rows)


def write_report(report_dir: Path, input_path: Path, features_path: Path, llm_path: Path,
                 matrix: pd.DataFrame, feature_columns: list[str], audit: dict[str, Any],
                 constants: list[str], options: argparse.Namespace, invalid_lines: int,
                 dropped_target: int, missing_extractions: int,
                 prompt_versions: list[str], dictionary: pd.DataFrame) -> None:
    """Write the Markdown report that documents the matrix."""
    row_count = len(matrix)
    families = family_table(audit, row_count)
    groups = group_table(dictionary, audit, row_count)
    levels = audit["levels"]
    unmapped_total = sum(audit["unmapped"].values())
    disagreements = pd.DataFrame([
        {"признак": name, "расхождений": count}
        for name, count in audit["disagreements"].items()
    ])
    highway_levels = dictionary.loc[dictionary["kind"].eq("one_hot")][["column", "note"]]

    features_total = len(feature_columns)
    binary = [column for column in feature_columns if matrix[column].nunique() <= 2]
    filled_versions = ", ".join(f"`{version}`" for version in prompt_versions) or "нет"

    report = [
        "# Компактная матрица: числовой каркас и 10 групп признаков",
        "",
        f"Источник: `{input_path}`",
        f"Извлечения из текста: `{llm_path}`",
        f"Матрица: `{features_path}`",
        f"Строк: **{counted(row_count)}**; колонок всего: **{matrix.shape[1]}** "
        f"(признаков **{features_total}**, плюс `{IDENTIFIER_COLUMN}`, цель "
        f"`{TARGET_COLUMN}`, вариант цели `{LOG_TARGET_COLUMN}` и ключ групп "
        f"`{GROUP_COLUMN}`).",
        f"Признаки: **{features_total - len(binary)}** числовых и **{len(binary)}** бинарных "
        "(категориальные уровни развёрнуты в колонки-индикаторы).",
        f"Битых строк JSON: **{invalid_lines}**; строк без положительной цены отброшено: "
        f"**{dropped_target}**.",
        f"Статусы извлечений, значения которых разрешено использовать: "
        f"{', '.join(f'`{status}`' for status in options.llm_statuses)}.",
        "",
        "## Группы признаков",
        "Числовой каркас берётся из карточки объявления (извлечения заполняют пропуски), "
        "остальные группы собираются из ответов модели. У каждой группы свой словарь "
        "уровней, поэтому синонимы не разъезжаются по разным колонкам.",
        markdown_table(groups),
        "",
        "## Признаки и источники значений",
        "Источник «карточка + LLM» означает, что значение карточки приоритетно и только "
        "пустые места заполнены извлечением.",
        markdown_table(families, max_rows=40),
        "",
        "## Канонические уровни групп",
        "Уровни описаны в скрипте (`CATEGORICAL_GROUPS`) и выводятся всегда, даже если "
        "уровень пуст: схема матрицы не зависит от выборки. Текст вне словаря попадает в "
        "`other` (для групп со свободным текстом) или в `unknown` вместе с отсутствующими "
        f"значениями; всего значений вне словаря: **{counted(unmapped_total)}**.",
        markdown_table(levels, max_rows=60),
        "",
        "## Уровни шоссе (one-hot по частоте)",
        f"Шоссе — часть числового каркаса: уровень получает отдельную колонку, если "
        f"встречается не реже {options.one_hot_min_count} раз; редкие уровни объединены "
        "в `_other`. Константные колонки шоссе удаляются автоматически.",
        markdown_table(highway_levels, max_rows=40),
        "",
        "## Расхождения карточки и извлечений",
        "Строки, где оба источника дали значение, но значения разошлись (для чисел — "
        "относительное расхождение больше 1 %). В матрицу попадает значение карточки.",
        markdown_table(disagreements),
        "",
        "## Статусы извлечений",
        "Промпт различает существующее, запланированное и возможное, поэтому значения "
        "со статусом `planned`/`possible` не используются: газ по границе — не "
        "подключённый газ. Набор статусов задаётся ключом `--llm-statuses`.",
        markdown_table(audit["statuses"], max_rows=40),
        "",
        f"Версии промпта в файле извлечений: {filled_versions} "
        f"(базовый промпт `prompts/prompt_v1.txt`, ожидаемая версия `{PROMPT_VERSION}`).",
        f"Строк датасета без ответа модели: **{missing_extractions}**.",
        "Ответы присоединяются к объявлениям по `listing_id`. Если выгрузка менялась "
        "после извлечения, значения могли устареть — тогда перезапустите "
        "`build_llm_extractions.py`, чтобы обновить кэш.",
        "",
        "## Особенности матрицы",
        "- Пропуски **не заполняются**: NaN остаётся, потому что "
        "`HistGradientBoostingRegressor`, XGBoost и LightGBM обрабатывают NaN сами, "
        "а заполнение до разбиения на фолды переносит информацию из теста в обучение. "
        "Для линейных моделей заполняйте NaN медианой, посчитанной **только на "
        "обучающей части** каждого фолда.",
        "- Флаги различают «нет» (0) и «нет данных» (NaN): у планировки 0 — явное "
        "отрицание в тексте (мансарды нет), у строений 0 — модель не нашла такое "
        "строение в описании, а NaN — ответа модели по объявлению нет вовсе.",
        "- Признаки, производные от цены (`price_per_m2_*`), не строятся, а метка "
        "платформы «Выгодная цена» не используется: она вычисляется из цены.",
        f"- Константные колонки, удалённые автоматически (только уровни шоссе): "
        f"{', '.join(f'`{name}`' for name in constants) or 'нет'}. Уровни групп остаются "
        "на месте, чтобы схема не менялась между прогонами.",
        "",
        "## Как обучать на этой матрице",
        "1. Целевая переменная: `log_asking_price_rub` для скошенного распределения — "
        "метрики затем пересчитывайте обратно через `expm1`.",
        "2. Модели: `HistGradientBoostingRegressor` или LightGBM/XGBoost — они нативно "
        "работают с NaN.",
        f"3. Разбиение: `GroupKFold` с группами из `{GROUP_COLUMN}` — объявления одного "
        "населённого пункта не должны попадать одновременно в обучение и тест.",
        "",
        "## Файлы",
        f"- Матрица признаков: `{features_path}`",
        f"- Словарь признаков: `{report_dir / 'feature_dictionary.csv'}`",
        f"- Покрытие и статистика: `{report_dir / 'feature_coverage.csv'}`",
        f"- Отчёт: `{report_dir / 'report.md'}`",
    ]
    (report_dir / "report.md").write_text("\n".join(report) + "\n", encoding="utf-8")


def run(input_path: Path, llm_path: Path, features_path: Path, report_dir: Path,
        options: argparse.Namespace) -> None:
    raw, invalid_lines = read_jsonl(input_path)
    target = numeric(raw, TARGET_COLUMN)
    keep = target.gt(0)
    dropped_target = int((~keep).sum())
    raw = raw.loc[keep].reset_index(drop=True)
    if raw.empty:
        raise SystemExit("Не осталось строк с положительной ценой.")

    records: dict[str, dict[str, Any]] = {}
    llm_stats: dict[str, int] = {"records": 0, "invalid_lines": 0, "duplicates": 0}
    if options.llm_fill:
        records, llm_stats = load_extractions(llm_path)
        if not records:
            LOG.warning("В %s нет ни одного ответа модели: группы останутся пустыми",
                        llm_path)
    else:
        LOG.warning("Извлечения отключены (--no-llm-fill): останется только карточка")
    features_by_id = extraction_features(records)
    buildings_by_id = extraction_buildings(records)

    builder, audit = build_features(raw, features_by_id, buildings_by_id, options)
    dictionary = builder.dictionary()
    features = builder.frame()

    # Constants are dropped only among the frequency-encoded highway levels; the
    # canonical group vocabulary keeps its columns so the schema stays stable.
    droppable = set(dictionary.loc[dictionary["kind"].eq("one_hot"), "column"])
    constants = [column for column in
                 constant_columns(features, (TARGET_COLUMN, LOG_TARGET_COLUMN))
                 if column in droppable]
    if constants:
        features = features.drop(columns=constants)
        dictionary = dictionary.loc[~dictionary["column"].isin(constants)]

    if IDENTIFIER_COLUMN in raw:
        listing_ids = raw[IDENTIFIER_COLUMN]
    else:
        listing_ids = pd.Series(np.arange(len(raw)))
    matrix = pd.concat(
        [listing_ids.rename(IDENTIFIER_COLUMN).reset_index(drop=True), features], axis=1
    )

    # Group key for GroupKFold: listings of the same locality must not be split across
    # folds, otherwise near-duplicate objects leak between train and test. Not a feature.
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
    known_ids = set(records)
    missing_extractions = int(sum(
        1 for listing_id in listing_ids if str(listing_id) not in known_ids
    ))
    prompt_versions = sorted({
        str(record.get("prompt_version")) for record in records.values()
        if record.get("prompt_version") is not None
    })
    write_report(report_dir, input_path, features_path, llm_path, matrix, feature_columns,
                 audit, constants, options, invalid_lines, dropped_target,
                 missing_extractions, prompt_versions, dictionary)

    print(f"Строк: {counted(len(matrix))}; колонок: {matrix.shape[1]} "
          f"(признаков {len(feature_columns)})")
    print(f"Заполнено из извлечений (каркас): "
          f"{sum(s['from_llm'] for name, s in audit['sources'].items() if name in STRUCTURED_FEATURES)} "
          f"значений; ответов модели прочитано: {llm_stats['records']}")
    print(f"Матрица признаков: {features_path}")
    print(f"Отчёт и словарь: {report_dir}")


def parse_args() -> argparse.Namespace:
    def statuses(value: str) -> tuple[str, ...]:
        parsed = tuple(item.strip().lower() for item in value.split(",") if item.strip())
        if not parsed:
            raise argparse.ArgumentTypeError(
                "Укажите хотя бы один статус, например stated."
            )
        return parsed

    parser = argparse.ArgumentParser(
        description="Построение компактной матрицы: числовой каркас из карточки и группы "
                    "признаков из извлечений текста, с каноническими уровнями."
    )
    parser.add_argument(
        "--input", type=Path, default=default_input_path(),
        help="Путь к objects.jsonl (по умолчанию data/dataset/objects.jsonl).",
    )
    parser.add_argument(
        "--llm-extractions", type=Path, default=default_llm_path(),
        help="Кэш ответов модели (по умолчанию data/features/llm_extractions.jsonl).",
    )
    parser.add_argument(
        "--features-file", type=Path, default=default_features_path(),
        help="Куда записать матрицу признаков (по умолчанию data/features/features_v1.csv).",
    )
    parser.add_argument(
        "--report-dir", type=Path, default=default_report_dir(),
        help="Каталог для отчёта и словаря (по умолчанию reports/features_v1).",
    )
    parser.add_argument(
        "--one-hot-min-count", type=int, default=10,
        help="Минимальное число строк, чтобы уровень шоссе получил свою колонку.",
    )
    parser.add_argument(
        "--llm-statuses", type=statuses, default=DEFAULT_LLM_STATUSES,
        help="Статусы извлечений, значения которых попадают в матрицу "
             f"(по умолчанию {','.join(DEFAULT_LLM_STATUSES)}).",
    )
    parser.add_argument(
        "--no-llm-fill", dest="llm_fill", action="store_false", default=True,
        help="Не читать llm_extractions.jsonl: останутся только поля карточки.",
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
        args.llm_extractions.expanduser(),
        args.features_file.expanduser(),
        args.report_dir.expanduser(),
        args,
    )


if __name__ == "__main__":
    main()
