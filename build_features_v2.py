#!/usr/bin/env python3
"""Build the compact feature matrix for house-price models: numeric backbone + categorical groups.

Source:  data/dataset/objects.jsonl — structured fields of the Domclick export.
         prompts/prompt_v2_cat.txt - for categorical features,
         prompts/prompt_v2_num.txt - for numerical features.
         ollama — local server (default `http://127.0.0.1:11434`) with the `qwen3.8` model;
         run `ollama serve` and `ollama pull qwen3.8` beforehand, or use --no-llm.
Output:  data/features/features_v2.csv — one row per listing, numeric only.
         data/features/llm_v2_extractions.jsonl — cache of the model answers, resumable.
         reports/features_v2/ — dictionary, coverage, extraction summary, levels, report.

Числовой каркас — из карточки объявления, пропуски заполняются извлечениями ollama qwen3.8:
    house_area_m2, distance_to_mkad_km, highway, land_area_sotka, floors,
    house_area_per_floor, land_to_house_ratio

Группы — собираются из извлечений, у каждой свой канонический словарь уровней:
    Земля ................ izhs (ИЖС + земли населенных пунктов) / snt (СНТ вместе с ДНП) / lph / other (сельхоз и прочие редкие категории)
    Отделка .............. with_repair (готовая или частичная) / without_repair (без отделки или предчистовая или неизвестно)
    Газ .................. подключён / по границе (рядом) / отсутствует (неизвестно)
    Стены ................ газобетон / пенобетон / кирпич / каркас / брус / бревно / другие
    Фундамент ............ плита / ленточный / сваи / свайно-ростверковый / другой
    Отопление ............ gazovoe (магистральное) / gazgolder (газгольдер) / elektricheskoe / pechnoe (печное или твердотопливное) / отсутствует (неизвестно)
    Вода ................. centralnoe / skvazhina / kolodec / отстутсвует (неизвестно)
    Канализация .......... centralnoe / septik	/ отстутсвует (выгребная яма, неизвестно)
    Планировка ........... кол-во спален, кол-во санузлов, есть_мансарда, есть_подвал
    Дополнительные строения ... есть_баня, есть_гараж, есть_гостевой дом и их площади
    Возраст .............. текущий год минус год постройки

Design rules
------------
1. The structured card is the primary source for the numeric backbone and for the land
   category (its card tags); the extraction fills their gaps. The groups themselves are
   built only from the extractions.
2. Every answer is cached per listing together with the hash of the advertisement text,
   the hash of both prompts, the schema hash and the model name, and the record is
   flushed + fsynced to disk before the next request: a power loss cannot lose finished
   listings, and a run can always be continued with the same command. The tail line of a
   cache cut short mid-write is repaired on the next start.
3. Free-text and enum answers are collapsed into the canonical vocabulary of their group
   ("газоблок", "газосиликат", "gazobeton" -> `gazobeton`); the vocabulary columns are
   always emitted, so the matrix schema does not depend on the sample.
4. Groups whose dictionary has no "other" level (газ, отопление, вода, канализация,
   отделка) collapse unknown answers into their negative level, exactly as the task
   defines it: "газ отсутствует (неизвестно)". Rows without any model answer stay NaN
   instead, so "не спрашивали" never turns into "нет газа".
5. Implausible numbers (a 400 m² barn, a 30-storey house) are dropped to NaN and counted
   in the report: they are model hallucinations, not measurements. The price and the
   platform badges are never passed to the model, because the extractions become
   features and the asking price is the target.
6. Missing values stay NaN instead of being imputed: HistGradientBoosting, XGBoost and
   LightGBM handle NaN natively, and imputing before a train/test split leaks
   information.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import re
import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd


try:  # ollama is optional: the card-only matrix (--no-llm) works without it.
    import ollama
except ImportError:  # pragma: no cover - depends on the environment
    ollama = None  # type: ignore[assignment]


LOG = logging.getLogger("domclick.features_v2")

IDENTIFIER_COLUMN = "listing_id"
TARGET_COLUMN = "asking_price_rub"
LOG_TARGET_COLUMN = "log_asking_price_rub"
GROUP_COLUMN = "group_locality"

# Carried for training and grouped validation, never used as features.
NON_FEATURE_COLUMNS = (IDENTIFIER_COLUMN, TARGET_COLUMN, LOG_TARGET_COLUMN, GROUP_COLUMN)

# Year used to turn the construction year into the age of the house.
CURRENT_YEAR = 2026

# Name of the column that holds the age; the year is recoverable as CURRENT_YEAR - age.
AGE_COLUMN = "house_age"

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
    "Вода",
    "Канализация",
    "Планировка",
    "Дополнительные строения",
    "Возраст",
)

# (kind, source, note) of the numeric backbone; the column order is fixed in BACKBONE_ORDER.
BACKBONE_META: dict[str, tuple[str, str, str]] = {
    "house_area_m2": (
        "numeric", "objects.jsonl:house_area_m2 | llm:house_area_m2",
        "площадь дома, м²; пропуски карточки заполнены извлечением из текста",
    ),
    "distance_to_mkad_km": (
        "numeric", "objects.jsonl:distance_to_mkad_km | llm:distance_to_mkad_km",
        "расстояние до МКАД, км; пустые значения заполнены извлечением",
    ),
    "highway": (
        "categorical", "objects.jsonl:highway | llm:highway",
        "шоссе; пустые значения заполнены извлечением, затем one-hot по частоте",
    ),
    "land_area_sotka": (
        "numeric", "objects.jsonl:land_area_sotka | llm:land_area_sotka",
        "площадь участка, соток; пропуски карточки заполнены извлечением",
    ),
    "floors": (
        "numeric", "objects.jsonl:floors | llm:floors",
        "этажность; пропуски карточки заполнены извлечением",
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

# Backbone families that objects.jsonl provides; the extraction only fills their gaps.
STRUCTURED_FEATURES = ("house_area_m2", "distance_to_mkad_km", "highway",
                       "land_area_sotka", "floors")

# Fields the backbone takes from the extraction.
BACKBONE_LLM_FIELDS: dict[str, str] = {
    "house_area_m2": "house_area_m2",
    "distance_to_mkad_km": "distance_to_mkad_km",
    "highway": "highway",
    "land_area_sotka": "land_area_sotka",
    "floors": "floors",
}

# Canonical categorical groups. `levels` are ordered rules (canonical level, regex);
# the first matching rule wins, so a narrower rule must come first. Every rule also
# matches the token the prompt returns (for example `^connected$`), so a model answer
# and a Russian phrase canonicalize into the same level.
#   catch_all     — level for recognised but unexpected text (None: it stays unknown)
#   unknown_level — level for absent/unknown values (the task defines where "unknown" goes)
CATEGORICAL_GROUPS: tuple[dict[str, Any], ...] = (
    {
        "group": "Земля",
        "prefix": "land_category",
        "card_field": "land_use_label",
        "llm_fields": ("land_category", "permitted_use"),
        "levels": (
            ("snt", r"^snt$|^dnp$|\bснт\b|\bднп\b|\bсдт\b|садовод|огороднич|дачн|"
                    r"коллективн\w* хозяйств"),
            ("lph", r"^lph$|\bлпх\b|личн\w* подсобн|приусадебн|подсобн\w* хозяйств"),
            ("izhs", r"^izhs$|\bижс\b|индивидуальн\w* жилищ|жилищн\w* строительств|"
                     r"жилой застройк|жилыми зданиями|малоэтажн\w* жил|"
                     r"земл\w* насел[её]нн|насел[её]нных пунктов|земл\w* поселени|\bзнп\b"),
        ),
        "catch_all": "other",
        "unknown_level": "unknown",
        "note": "категория земель и вид разрешённого использования: izhs (ИЖС и земли "
                "населённых пунктов) / snt (СНТ и ДНП) / lph / other (сельхоз и прочие "
                "редкие категории); источник — метки карточки, затем извлечение",
    },
    {
        "group": "Отделка",
        "prefix": "finishing_stage",
        "field": "finishing_stage",
        "levels": (
            # Отрицание проверяется первым: «без отделки» и «требует ремонта» содержат
            # те же слова, что и «готовая отделка».
            ("without_repair", r"^without_repair$|без отделки|без ремонта|чернов|предчистов|"
                               r"требует ремонта|не отремонтир|под отделку|незаверш[её]н|"
                               r"недострой"),
            ("with_repair", r"^with_repair$|готова|с отделкой|чистовая|отделк|отремонтир|"
                            r"ремонт|заезжай и живи|меблирован"),
        ),
        "catch_all": None,
        "unknown_level": "without_repair",
        "note": "отделка: with_repair (готовая или частичная) / without_repair (без "
                "отделки, предчистовая или неизвестно)",
    },
    {
        "group": "Газ",
        "prefix": "gas_status",
        "field": "gas_status",
        "levels": (
            ("po_granice", r"^at_boundary$|границ|врезк|рядом с участком|газ\w* рядом"),
            ("podklyuchen", r"^connected$|подключ|подвед[её]н|есть газ|газ есть|"
                            r"газифицирован|магистральн\w* газ"),
        ),
        "catch_all": None,
        "unknown_level": "otsutstvuet",
        "note": "газ: po_granice / podklyuchen / otsutstvuet (нет газа или неизвестно)",
    },
    {
        "group": "Стены",
        "prefix": "wall_material",
        "field": "wall_material",
        "levels": (
            ("gazobeton", r"^gazobeton$|газобетон|газоблок|газосиликат|zobeton|"
                          r"автоклавн"),
            ("penobeton", r"^penobeton$|пенобетон|пеноблок"),
            ("kirpich", r"^kirpich$|кирпич"),
            ("karkas", r"^karkas$|каркас|сип|\bsip\b|сэндвич|щитов"),
            ("brus", r"^brus$|брус"),
            ("brevno", r"^brevno$|бревн|бревен|оцилиндрован|сруб|лафет"),
        ),
        "catch_all": "other",
        "unknown_level": "unknown",
        "note": "стены: газобетон / пенобетон / кирпич / каркас / брус / бревно / другие; "
                "«дерево», керамзитобетон и блоки без уточнения попадают в «другие»",
    },
    {
        "group": "Фундамент",
        "prefix": "foundation_type",
        "field": "foundation_type",
        "levels": (
            ("svayno_rostverkovyy", r"^svayno_rostverkovyy$|ростверк"),
            ("lentochny", r"^lentochny$|ленточн|\bлента\b|фбс|\bfbs\b"),
            ("plita", r"^plita$|плит|ушп|монолит"),
            ("svai", r"^svai$|сва[йия]|винтов|буронабивн"),
        ),
        "catch_all": "other",
        "unknown_level": "unknown",
        "note": "фундамент: плита / ленточный / сваи / свайно-ростверковый / другой",
    },
    {
        "group": "Отопление",
        "prefix": "heating_type",
        "field": "heating_type",
        "levels": (
            ("gazgolder", r"^gazgolder$|газгольдер|сжиженн|сжиж"),
            ("gazovoe", r"^gazovoe$|магистральн\w* газ|газов\w* кот|газовое отопление|"
                        r"отоплен\w* газ"),
            ("elektricheskoe", r"^elektricheskoe$|электр|конвектор|обогрев|инфракрасн|"
                               r"теплов[оа]й насос"),
            ("pechnoe", r"^pechnoe$|печ|камин|твердотопливн|дров|уголь|пеллет"),
        ),
        "catch_all": None,
        "unknown_level": "otsutstvuet",
        "note": "отопление: gazovoe (магистральный газ) / gazgolder (газгольдер) / "
                "elektricheskoe / pechnoe (печное или твердотопливное) / otsutstvuet "
                "(нет отопления или неизвестно)",
    },
    {
        "group": "Вода",
        "prefix": "water_source",
        "field": "water_source",
        "levels": (
            # Привозная и летняя вода — не центральное водоснабжение: словарь относит
            # такие ответы к уровню «отсутствует» вместе с неизвестными.
            ("otsutstvuet", r"^absent$|^unknown$|^other$|летн|привозн|завоз|нет воды|"
                            r"отсутств|нет водоснабжен"),
            ("skvazhina", r"^skvazhina$|скважин|абиссин|артезиан"),
            ("kolodec", r"^kolodec$|колодец|колодц"),
            ("centralnoe", r"^centralnoe$|центральн|централизован|водопровод|городск|"
                           r"магистральн|поселков"),
        ),
        "catch_all": None,
        "unknown_level": "otsutstvuet",
        "note": "вода: centralnoe / skvazhina / kolodec / otsutstvuet (нет воды, "
                "привозная или летняя вода, неизвестно)",
    },
    {
        "group": "Канализация",
        "prefix": "sewage_type",
        "field": "sewage_type",
        "levels": (
            ("septik", r"^septik$|септик|септ|био|топас|юнилос|автономн|локальн|"
                       r"очистк|\bлос\b"),
            ("centralnoe", r"^centralnoe$|центральн|централизован|поселков"),
        ),
        "catch_all": None,
        "unknown_level": "otsutstvuet",
        "note": "канализация: centralnoe / septik (в том числе станции биоочистки) / "
                "otsutstvuet (выгребная яма, нет канализации или неизвестно)",
    },
)

# (group, column, extraction field, kind, note); kind ∈ {count, flag}.
NUMBER_FEATURES: tuple[tuple[str, str, str, str, str], ...] = (
    ("Планировка", "bedrooms", "bedrooms_count", "count", "число спален"),
    ("Планировка", "bathrooms", "bathrooms_count", "count", "число санузлов"),
    ("Планировка", "attic_present", "has_attic", "flag", "заявлена мансарда"),
    ("Планировка", "basement_present", "has_basement", "flag",
     "заявлен подвал или цокольный этаж"),
)

# Additional buildings: a presence flag and the area of each type.
BUILDING_GROUP = "Дополнительные строения"
BUILDING_SPECS: tuple[tuple[str, str, str, str], ...] = (
    ("has_bathhouse", "bathhouse_area", "bathhouse", "баня или сауна"),
    ("has_garage", "garage_area", "garage", "гараж"),
    ("has_guesthouse", "guesthouse_area", "guest_house", "гостевой дом"),
)

# Column order of the group part of the matrix: families, not data types.
EMIT_ORDER: tuple[str, ...] = (
    "land_category",
    "finishing_stage",
    "gas_status",
    "wall_material",
    "foundation_type",
    "heating_type",
    "water_source",
    "sewage_type",
    "bedrooms", "bathrooms", "attic_present", "basement_present",
    "bathhouse_present", "bathhouse_area_m2",
    "garage_present", "garage_area_m2", "guest_house_present", "guest_house_area_m2",
    AGE_COLUMN,
)

# Plausible range of every numeric extraction: everything else is a model hallucination.
VALUE_RANGES: dict[str, tuple[float, float]] = {
    "house_area_m2": (5.0, 20_000.0),
    "land_area_sotka": (0.1, 5_000.0),
    "floors": (1.0, 10.0),
    "distance_to_mkad_km": (0.0, 500.0),
    "bedrooms": (1.0, 20.0),
    "bathrooms": (1.0, 20.0),
    "bathhouse_area_m2": (1.0, 1_000.0),
    "garage_area_m2": (1.0, 1_000.0),
    "guest_house_area_m2": (1.0, 2_000.0),
    AGE_COLUMN: (0.0, 200.0),
}

# Level that collects absent values. Never written literally into a group: every group of
# CATEGORICAL_GROUPS declares its own `unknown_level`, which is either this level (when the
# group also has an `other` level for unrecognised text) or a negative level the unknown
# value collapses into.
UNKNOWN_LEVEL = "unknown"

# Plausible range for the year fields: everything else is a data error, not a year.
YEAR_RANGE = (1850.0, float(CURRENT_YEAR + 1))

# Tokens that mean True in a boolean extraction value (mirrors to_flag).
TRUE_TOKENS = frozenset({"true", "да", "yes", "есть", "1"})

# Values that mean "the field is empty" rather than a level of the characteristic.
EMPTY_TEXT_VALUES = frozenset({"", "-", "none", "null", "нет", "нет данных", "не указано",
                               "не указана", "unknown", "неизвестно"})

# Placeholder of the advertisement inside both prompts.
PROMPT_PLACEHOLDER = "{{description}}"
PROMPT_VERSION = "2.0"

DEFAULT_MODEL = "qwen3.8"
DEFAULT_HOST = "http://127.0.0.1:11434"

# Measured pace of a local model on this dataset (two prompts per listing): the estimate in
# the run header is only a hint, the real speed is printed with every progress line.
SECONDS_PER_LISTING = 20.0
SECONDS_PER_LISTING_THINK = 60.0

# Card fields rendered into the advertisement block. The price and the platform badges are
# deliberately absent: the extractions become features, so feeding the target (or a badge
# computed from it) to the model would leak the price into the matrix.
ADVERTISEMENT_FIELDS: tuple[tuple[str, str], ...] = (
    ("listing_url", "Ссылка"),
    ("object_type", "Тип объекта"),
    ("locality", "Населённый пункт"),
    ("settlement", "Посёлок или СНТ"),
    ("address_fragment", "Адрес"),
    ("land_use_label", "Вид использования земли"),
    ("tags_seller_claimed", "Метки продавца"),
)

# Card fields shown with their units in the advertisement block.
ADVERTISEMENT_NUMBERS: tuple[tuple[str, str, str], ...] = (
    ("highway", "Шоссе", ""),
    ("distance_to_mkad_km", "Расстояние до МКАД", "км"),
    ("house_area_m2", "Площадь дома по карточке", "м²"),
    ("land_area_sotka", "Площадь участка по карточке", "соток"),
    ("floors", "Этажей по карточке", ""),
)

# Words cut off before the highway is one-hot encoded, so that the card value
# ("Варшавское шоссе") and the model answer land in the same level.
HIGHWAY_NOISE = re.compile(
    r"\bш\.?\b|\bшоссе\b|\bшоссейн\w*\b|\bнаправление\b|\bтрасса\b|\bавтодорога\b",
    re.IGNORECASE,
)


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
                 source: str, note: str = "", answered: pd.Series | None = None) -> None:
        """Add a 0/1 column; rows without any model answer stay NaN."""
        values = mask.fillna(False).astype("float64")
        if answered is not None:
            values = values.where(answered.fillna(False))
        self.add(name, values, group, kind, source, note)

    def frame(self) -> pd.DataFrame:
        return pd.DataFrame(self.columns)

    def dictionary(self) -> pd.DataFrame:
        return pd.DataFrame(self.metadata)


# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #

def project_root() -> Path:
    return Path(__file__).resolve().parent


def default_input_path() -> Path:
    """Find the generated dataset in the usual project output locations."""
    root = project_root()
    candidates = (root / "data" / "dataset" / "objects.jsonl", )
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return candidates[0]


def default_prompt_cat_path() -> Path:
    return project_root() / "prompts" / "prompt_v2_cat.txt"


def default_prompt_num_path() -> Path:
    return project_root() / "prompts" / "prompt_v2_num.txt"


def default_cache_path() -> Path:
    """Cache of the raw answers; keeps the local-model pass resumable."""
    return project_root() / "data" / "features" / "llm_v2_extractions.jsonl"


def default_features_path() -> Path:
    """The compact matrix belongs in data/, next to the other feature matrices."""
    return project_root() / "data" / "features" / "features_v2.csv"


def default_report_dir() -> Path:
    """Reports belong in reports/, separate from the data."""
    return project_root() / "reports" / "features_v2"


# --------------------------------------------------------------------------- #
# Small helpers
# --------------------------------------------------------------------------- #

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
                    LOG.warning("Строка %s содержит JSON не в виде объекта; пропущена",
                                line_number)
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


def python_strings(values: pd.Series) -> pd.Series:
    """Same values on Python-backed strings.

    pandas 3 keeps text in a pyarrow-backed `string` column, and its vectorized regex
    uses RE2, where `\\b` and `\\w` are ASCII-only: a pattern like `\\bижс\\b` silently
    never matches Cyrillic. Matching therefore runs on object dtype, which uses `re`.
    """
    return values.astype(object)


def clean_categorical(values: pd.Series) -> pd.Series:
    """Normalized categorical text: lowercase, collapsed spaces, no trailing punctuation."""
    cleaned = values.astype("string").fillna("").str.strip().str.lower()
    cleaned = cleaned.str.replace("\u00a0", " ", regex=False)
    cleaned = cleaned.str.replace(r"\s+", " ", regex=True)
    cleaned = cleaned.str.replace(r"[.,;]+$", "", regex=True)
    return cleaned.mask(cleaned.isin(EMPTY_TEXT_VALUES))


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


def canonical_highway(values: pd.Series) -> pd.Series:
    """Highway name without boilerplate, so card and model values share one level."""
    cleaned = clean_categorical(values)
    stripped = python_strings(cleaned).str.replace(HIGHWAY_NOISE, " ", regex=True)
    stripped = stripped.str.replace(r"\s+", " ", regex=True).str.strip()
    # An empty remainder means the value was pure boilerplate ("шоссе") — keep the original.
    return stripped.mask(stripped.eq(""), python_strings(cleaned))


def counted(value: int) -> str:
    """Group thousands with a space, the way the other reports do."""
    return f"{value:,}".replace(",", " ")


def human_duration(seconds: float) -> str:
    """Duration in words for the console: "45 с", "12 мин", "2 ч 05 мин"."""
    seconds = max(float(seconds), 0.0)
    if seconds < 90:
        return f"{seconds:.0f} с"
    minutes = seconds / 60.0
    if minutes < 90:
        return f"{minutes:.0f} мин"
    hours, rest = divmod(int(round(minutes)), 60)
    return f"{hours} ч {rest:02d} мин"


def finish_clock(seconds: float) -> str:
    """Wall clock time at which the remaining work is expected to be over."""
    return (datetime.now() + timedelta(seconds=max(float(seconds), 0.0))).strftime("%H:%M")


def cache_size_bytes(path: Path) -> float:
    """Size of the answer cache on disk, in bytes."""
    try:
        return float(path.stat().st_size)
    except OSError:
        return 0.0


def size_text(num_bytes: float) -> str:
    """Human-readable size for the console: "420 Б", "3 КБ", "1.4 МБ"."""
    if num_bytes < 1024:
        return f"{num_bytes:.0f} Б"
    if num_bytes < 1024 * 1024:
        return f"{num_bytes / 1024:.0f} КБ"
    return f"{num_bytes / 1024 / 1024:.1f} МБ"


def plural(count: int, one: str, few: str, many: str) -> str:
    """Russian plural form: plural(2, "строка", "строки", "строк") -> "строки"."""
    tail = abs(int(count)) % 100
    if 11 <= tail <= 14:
        return many
    last = tail % 10
    if last == 1:
        return one
    if last in (2, 3, 4):
        return few
    return many


def cache_count_text(path: Path) -> str:
    """Records on disk with the right plural and size: "24 записи (11 КБ)"."""
    lines = cache_line_count(path)
    return f"{counted(lines)} {plural(lines, 'запись', 'записи', 'записей')} " \
           f"({size_text(cache_size_bytes(path))})"


def cache_line_count(path: Path) -> int:
    """Non-empty lines of the cache: every line is one saved answer."""
    if not path.is_file():
        return 0
    try:
        with path.open("r", encoding="utf-8-sig") as stream:
            return sum(1 for line in stream if line.strip())
    except OSError:
        return 0


def ensure_trailing_newline(path: Path) -> bool:
    """Repair a tail line that a power loss left without a newline.

    A record is written as a single line, but a crash can still cut the last line short;
    appending straight away would merge it with the next record and lose both. Returns
    True when the file had to be repaired.
    """
    try:
        if not path.is_file() or path.stat().st_size == 0:
            return False
        with path.open("rb") as stream:
            stream.seek(-1, os.SEEK_END)
            if stream.read(1) == b"\n":
                return False
    except OSError as exc:
        LOG.warning("Не удалось проверить конец кэша %s: %s", path, exc)
        return False
    try:
        with path.open("a", encoding="utf-8") as stream:
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
    except OSError as exc:
        LOG.warning("Не удалось дописать перевод строки в кэш %s: %s", path, exc)
        return False
    return True


def say(message: str = "") -> None:
    """Console line for a long run: flushed right away, so it survives an interruption."""
    print(console_safe(message), flush=True)


def progress_message(*, done: int, total: int, elapsed: float, failures: int,
                     cache_path: Path) -> str:
    """One console line about the pass: pace, elapsed time, ETA and cache state."""
    pace = done / elapsed if elapsed > 0 else 0.0
    left = (total - done) / pace if pace > 0 else 0.0
    return (f"Прогресс: {counted(done)}/{counted(total)} ({done / total * 100:.1f}%); "
            f"скорость {pace * 60:.1f} объявл./мин; прошло {human_duration(elapsed)}; "
            f"осталось ~{human_duration(left)} (к {finish_clock(left)}); "
            f"ошибок {counted(failures)}; кэш {size_text(cache_size_bytes(cache_path))}")


def console_safe(value: str) -> str:
    """Console-safe text: the Windows console is cp1251 and rejects ₽, ², ± and −."""
    return value.replace("₽", "руб.").replace("²", "2").replace("±", "+-").replace("−", "-")


# --------------------------------------------------------------------------- #
# Prompts
# --------------------------------------------------------------------------- #

def load_prompt(path: Path) -> str:
    """Read a prompt template and make sure it has the advertisement placeholder."""
    try:
        template = path.read_text(encoding="utf-8-sig")
    except OSError as exc:
        raise SystemExit(f"Не удалось прочитать промпт {path}: {exc}") from exc
    if PROMPT_PLACEHOLDER not in template:
        raise SystemExit(f"В промпте {path} нет плейсхолдера {PROMPT_PLACEHOLDER}.")
    return template


def render_prompt(template: str, advertisement: str) -> str:
    """Substitute the advertisement without interpreting braces of the JSON schema."""
    return template.replace(PROMPT_PLACEHOLDER, advertisement)


def prompt_hash(*templates: str) -> str:
    """Identity of the extraction prompt; changing a prompt invalidates the cache."""
    digest = hashlib.sha256()
    for template in templates:
        digest.update(template.encode("utf-8"))
        digest.update(b"\0")
    return digest.hexdigest()[:16]


def advertisement_text(record: dict[str, Any]) -> str:
    """Render one listing as the advertisement block sent to the model."""

    def render(value: Any) -> str:
        if value in (None, "", []):
            return "не указано"
        if isinstance(value, list):
            return "; ".join(str(entry) for entry in value)
        if isinstance(value, float) and value.is_integer():
            return str(int(value))
        return str(value).strip()

    lines = [f"{label}: {render(record.get(key))}" for key, label in ADVERTISEMENT_FIELDS]
    lines += [
        f"{label}, {unit}: {render(record.get(key))}" if unit
        else f"{label}: {render(record.get(key))}"
        for key, label, unit in ADVERTISEMENT_NUMBERS
    ]
    lines += ["", "Описание:", str(record.get("description") or "не указано")]
    return "\n".join(lines)


def text_hash(advertisement: str) -> str:
    return hashlib.sha256(advertisement.encode("utf-8")).hexdigest()[:16]


# --------------------------------------------------------------------------- #
# Ollama
# --------------------------------------------------------------------------- #

# Fields the matrix may take from an answer; anything else the model invents is dropped.
EXPECTED_FIELDS: frozenset[str] = frozenset(
    [field for _group, _column, field, _kind, _note in NUMBER_FEATURES]
    + [flag for flag, _area, _name, _note in BUILDING_SPECS]
    + [area for _flag, area, _name, _note in BUILDING_SPECS]
    + ["construction_year", AGE_COLUMN]
    + list(BACKBONE_LLM_FIELDS.values())
    + [field for spec in CATEGORICAL_GROUPS
       for field in ([spec["field"]] if spec.get("field") else list(spec["llm_fields"]))]
)

# Identity of the extraction contract: the prompt and the schema of an answer. A cached
# answer is reused only when both match, so extending the contract (or changing how answers
# are sanitized) forces a re-request instead of silently reusing answers cut to the older
# shape by `sanitize_extraction`.
SCHEMA_SHA256 = hashlib.sha256(
    "\n".join(sorted(EXPECTED_FIELDS)).encode("utf-8")
).hexdigest()[:8]


def ollama_client_factory(host: str) -> Any:
    """Client constructor that fails with a readable message when ollama is missing."""
    if ollama is None:
        raise SystemExit(
            "Пакет ollama не установлен: выполните `pip install ollama` "
            "или запустите скрипт с ключом --no-llm (тогда заполнятся только карточки)."
        )
    return ollama.Client(host=host)


def model_names(client: Any) -> list[str]:
    """Names of the models the local server has pulled."""
    try:
        listing = client.list()
    except Exception as exc:  # ollama raises ConnectionError/ResponseError
        raise SystemExit(
            f"Локальный сервер Ollama недоступен ({type(exc).__name__}: {exc}).\n"
            "Запустите `ollama serve` и повторите; матрицу без извлечений можно собрать "
            "с ключом --no-llm."
        ) from exc
    items = (listing.get("models", []) if isinstance(listing, dict)
             else getattr(listing, "models", []))
    names: list[str] = []
    for item in items or []:
        name = item.get("model") if isinstance(item, dict) else getattr(item, "model", None)
        if name:
            names.append(str(name))
    return names


def ensure_model(client: Any, model: str) -> None:
    """Fail fast when the requested model is not available locally."""
    names = model_names(client)
    if any(name == model or name.split(":")[0] == model for name in names):
        return
    raise SystemExit(
        f"Модель {model} не найдена в Ollama. Доступные модели: "
        f"{', '.join(names) if names else 'нет'}.\nВыполните `ollama pull {model}`."
    )


def message_content(response: Any) -> str:
    """Text of the assistant message, regardless of the client's return type."""
    message = (response.get("message") if isinstance(response, dict)
               else getattr(response, "message", None))
    if message is None:
        return ""
    content = (message.get("content") if isinstance(message, dict)
               else getattr(message, "content", ""))
    return content if isinstance(content, str) else ""


def parse_json_object(content: str) -> dict[str, Any] | None:
    """Extract one JSON object from a model answer, tolerating fences and prose."""
    if not isinstance(content, str):
        return None
    body = content.strip()
    if "<｜end▁of▁thinking｜>" in body:  # reasoning models wrap the answer into a thinking block
        body = body.split("<｜end▁of▁thinking｜>", 1)[1].strip()
    if body.startswith("```"):
        body = re.sub(r"^```[a-zA-Z]*\s*", "", body)
        body = re.sub(r"\s*```$", "", body).strip()
    candidates = [body]
    if "{" in body and "}" in body:
        candidates.append(body[body.find("{"):body.rfind("}") + 1])
    for candidate in candidates:
        if not candidate:
            continue
        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            return parsed
    return None


def sanitize_extraction(parsed: dict[str, Any]) -> dict[str, Any]:
    """Keep only the expected fields and drop nested structures."""
    result: dict[str, Any] = {}
    for field in EXPECTED_FIELDS:
        if field not in parsed:
            continue
        value = parsed[field]
        if isinstance(value, (dict, list, tuple, set)):
            continue
        result[field] = value
    return result


class OllamaExtractor:
    """One listing -> a validated JSON answer, requested per thread and cached upstream."""

    def __init__(self, host: str, model: str, prompt_cat: str, prompt_num: str,
                 think: bool, num_ctx: int, retries: int) -> None:
        self.host = host
        self.model = model
        self.prompt_cat = prompt_cat
        self.prompt_num = prompt_num
        self.think = think
        self.num_ctx = num_ctx
        self.retries = retries
        self._local = threading.local()

    def client(self) -> Any:
        client = getattr(self._local, "client", None)
        if client is None:
            client = ollama_client_factory(self.host)
            self._local.client = client
        return client

    def ask(self, template: str, advertisement: str) -> tuple[dict[str, Any] | None, str | None]:
        """Ask one prompt; returns the parsed answer or a description of the failure."""
        prompt = render_prompt(template, advertisement)
        error: str | None = None
        for attempt in range(self.retries + 1):
            try:
                response = self.client().chat(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    format="json",
                    think=self.think,
                    options={"temperature": 0.0, "num_ctx": self.num_ctx},
                )
            except Exception as exc:
                error = f"{type(exc).__name__}: {exc}"
                if attempt < self.retries:
                    time.sleep(1.0 + attempt)
                    continue
                return None, error
            parsed = parse_json_object(message_content(response))
            if parsed is not None:
                return sanitize_extraction(parsed), None
            error = "ответ не разобрался как JSON"
        return None, error

    def extract(self, advertisement: str) -> dict[str, Any]:
        """Both prompts for one listing, merged into a single flat answer."""
        started = time.perf_counter()
        cat, cat_error = self.ask(self.prompt_cat, advertisement)
        num, num_error = self.ask(self.prompt_num, advertisement)
        errors = {name: text for name, text in (("cat", cat_error), ("num", num_error)) if text}
        return {
            "extraction": {**(cat or {}), **(num or {})},
            "blocks": {"cat": cat or {}, "num": num or {}},
            "errors": errors,
            "elapsed_s": round(time.perf_counter() - started, 2),
        }


def load_cache(path: Path, *, model: str, prompts: str,
               schema: str) -> dict[str, dict[str, Any]]:
    """Read the answer cache and keep only records that are still valid."""
    if not path.is_file():
        return {}
    records: dict[str, dict[str, Any]] = {}
    invalid = 0
    try:
        with path.open("r", encoding="utf-8-sig") as stream:
            for line in stream:
                if not line.strip():
                    continue
                try:
                    item = json.loads(line)
                except json.JSONDecodeError:
                    invalid += 1
                    continue
                if not isinstance(item, dict) or item.get("listing_id") is None:
                    invalid += 1
                    continue
                if (item.get("model") != model or item.get("prompt_sha256") != prompts
                        or item.get("schema_sha256") != schema):
                    continue
                records[str(item["listing_id"])] = item
    except OSError as exc:
        raise SystemExit(f"Не удалось прочитать кэш извлечений {path}: {exc}") from exc
    if invalid:
        LOG.warning("В кэше %s пропущено %s %s", path, invalid,
                    plural(invalid, "повреждённая строка", "повреждённые строки",
                           "повреждённых строк"))
    return records


def append_cache(path: Path, record: dict[str, Any]) -> None:
    """Append one answer to the cache and push it to disk before returning.

    The record is one line, flushed and fsynced right away: a power loss cannot lose
    finished listings, and the next run reads them back instead of asking the model again.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(record, ensure_ascii=False) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def collect_extractions(raw: pd.DataFrame, cache_path: Path, options: argparse.Namespace,
                        ) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    """Fill the cache for every listing: cached answers are reused, the rest are requested."""
    indexed = raw.reset_index(drop=True)
    ids = [str(value) for value in indexed[IDENTIFIER_COLUMN]]
    advertisements = [advertisement_text(record) for record in indexed.to_dict("records")]
    stats: dict[str, Any] = {
        "requested": 0, "from_cache": 0, "failed": 0, "failures": {}, "seconds": 0.0,
        "prompt_sha256": None, "records": {},
    }

    if not options.llm_fill:
        say("Извлечения отключены (--no-llm): матрица соберётся только из полей карточки, "
            "колонки групп останутся пустыми.")
        return {}, stats

    prompt_cat = load_prompt(options.prompt_cat)
    prompt_num = load_prompt(options.prompt_num)
    digest = prompt_hash(prompt_cat, prompt_num)
    stats["prompt_sha256"] = digest

    repaired = ensure_trailing_newline(cache_path)
    cached = {} if options.refresh else load_cache(cache_path, model=options.model,
                                                   prompts=digest, schema=SCHEMA_SHA256)
    records: dict[str, dict[str, Any]] = dict(cached)
    stats["from_cache"] = len(cached)

    say(f"Кэш: {cache_path}")
    say(f"      на диске {cache_count_text(cache_path)}; актуальных ответов для этой "
        f"выборки: {counted(len(cached))}")
    if repaired:
        say("      последняя строка кэша была без перевода строки (прошлый запуск прерван "
            "на записи) — файл починен.")
    if options.refresh:
        say("      --refresh: кэш игнорируется, все объявления запрашиваются заново.")

    pending = [
        (listing_id, advertisement, text_hash(advertisement))
        for listing_id, advertisement in zip(ids, advertisements)
        if not (listing_id in cached
                and cached[listing_id].get("text_sha256") == text_hash(advertisement))
    ]
    if not pending:
        say(f"Актуальные ответы есть по всем {counted(len(ids))} "
            f"{plural(len(ids), 'объявлению', 'объявлениям', 'объявлениям')}: запросов к "
            f"модели не требуется, сразу собираю матрицу.")
        stats["records"] = records
        stats["missing"] = 0
        return records, stats

    extractor = OllamaExtractor(
        host=options.host, model=options.model, prompt_cat=prompt_cat, prompt_num=prompt_num,
        think=options.think, num_ctx=options.num_ctx, retries=options.retries,
    )
    ensure_model(extractor.client(), options.model)
    per_listing = SECONDS_PER_LISTING_THINK if options.think else SECONDS_PER_LISTING
    estimate = len(pending) * per_listing / max(options.parallel, 1)

    say(f"Модель: {options.model} на {options.host}; рассуждения: "
        f"{'включены' if options.think else 'выключены'}; параллельно: "
        f"{max(options.parallel, 1)} объявл.")
    say(f"Промпты: {options.prompt_cat.name} + {options.prompt_num.name} "
        f"(sha {digest}); версия промпта {PROMPT_VERSION}, схема ответа {SCHEMA_SHA256}")
    say(f"Объявлений в выборке: {counted(len(ids))}; берётся из кэша: {counted(len(cached))}; "
        f"нужно запросить: {counted(len(pending))} "
        f"{plural(len(pending), 'объявление', 'объявления', 'объявлений')} "
        f"(по 2 запроса на объявление)")
    say(f"Оценка времени: ~{human_duration(estimate)} (до "
        f"{finish_clock(estimate)}). Каждый готовый ответ сразу пишется на диск, поэтому "
        f"при отключении питания прогон продолжается с этого места.")
    if options.log_every:
        if options.log_every == 1:
            say("Прогресс: строка после каждого объявления (--log-every 0 отключает их).")
        else:
            say(f"Прогресс: строка каждые {options.log_every} объявлений "
                f"(--log-every 0 отключает их).")
    say("")

    started = time.perf_counter()
    failures: dict[str, str] = {}

    def handle(listing_id: str, hashed: str, answer: dict[str, Any]) -> None:
        record = {
            "listing_id": listing_id,
            "model": options.model,
            "prompt_version": PROMPT_VERSION,
            "prompt_sha256": digest,
            "schema_sha256": SCHEMA_SHA256,
            "text_sha256": hashed,
            "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            **answer,
        }
        append_cache(cache_path, record)
        records[listing_id] = record
        stats["requested"] += 1
        if answer["errors"]:
            failures.setdefault(listing_id,
                                "; ".join(f"{key}: {value}"
                                          for key, value in answer["errors"].items()))
        done = stats["requested"]
        if options.verbose:
            filled = sum(1 for value in (answer.get("extraction") or {}).values()
                         if value is not None)
            say(f"  [{counted(done)}/{counted(len(pending))}] {listing_id}: "
                f"{answer.get('elapsed_s', 0.0):.1f} с, заполненных полей {filled}"
                + (f", ошибка: {failures[listing_id]}" if listing_id in failures else ""))
        # Первая строка — сразу после первого объявления: по ней видно реальную скорость,
        # дальше прогресс пишется каждые `--log-every` объявлений.
        if options.log_every and (done == 1 or done % options.log_every == 0
                                  or done == len(pending)):
            say(progress_message(done=done, total=len(pending),
                                 elapsed=time.perf_counter() - started,
                                 failures=len(failures), cache_path=cache_path))

    try:
        if options.parallel > 1:
            with ThreadPoolExecutor(max_workers=options.parallel) as pool:
                futures: dict[Future[dict[str, Any]], tuple[str, str]] = {
                    pool.submit(extractor.extract, advertisement): (listing_id, hashed)
                    for listing_id, advertisement, hashed in pending
                }
                try:
                    for future in as_completed(futures):
                        listing_id, hashed = futures[future]
                        handle(listing_id, hashed, future.result())
                except KeyboardInterrupt:
                    for future in futures:
                        future.cancel()
                    raise
        else:
            for listing_id, advertisement, hashed in pending:
                handle(listing_id, hashed, extractor.extract(advertisement))
    except KeyboardInterrupt:
        say("")
        say(f"Прервано. Прогресс уже на диске: {cache_count_text(cache_path)} в "
            f"{cache_path}.")
        say("Продолжить с этого места: запустите ту же команду (без --refresh).")
        raise

    stats["seconds"] = time.perf_counter() - started
    stats["failed"] = len(failures)
    stats["failures"] = failures
    stats["records"] = records
    stats["missing"] = sum(1 for listing_id in ids if listing_id not in records)
    say("")
    say(f"Запросов сделано: {counted(stats['requested'])} "
        f"{plural(stats['requested'], 'запрос', 'запроса', 'запросов')} за "
        f"{human_duration(stats['seconds'])} "
        f"({stats['requested'] / max(stats['seconds'], 1) * 60:.1f} объявл./мин); "
        f"ошибок ответа: {counted(stats['failed'])}")
    say(f"В кэше теперь {cache_count_text(cache_path)}")
    if failures:
        sample = ", ".join(list(failures)[:3])
        LOG.warning("Без корректного ответа осталось %s объявлений; пример: %s -> %s",
                    len(failures), next(iter(failures)), next(iter(failures.values())))
        say(f"Без корректного ответа: {counted(len(failures))} "
            f"{plural(len(failures), 'объявление', 'объявления', 'объявлений')} "
            f"(например, {sample}); их признаки групп останутся NaN в матрице.")
    if stats["missing"]:
        say(f"Без ответа модели осталось {counted(stats['missing'])} "
            f"{plural(stats['missing'], 'объявление', 'объявления', 'объявлений')}: "
            f"перезапустите ту же команду, чтобы добрать — готовые ответы не "
            f"запрашиваются повторно.")
    return records, stats


# --------------------------------------------------------------------------- #
# Extractions -> canonical levels and numbers
# --------------------------------------------------------------------------- #

def extraction_values(records: dict[str, dict[str, Any]], ids: pd.Series,
                      field: str) -> pd.Series:
    """Raw value of one extraction field per listing (None when the field is absent)."""
    values: list[Any] = []
    for listing_id in ids:
        record = records.get(str(listing_id)) or {}
        extraction = record.get("extraction")
        block = extraction if isinstance(extraction, dict) else {}
        values.append(block.get(field))
    return pd.Series(values, index=ids.index, dtype="object")


def answered_mask(records: dict[str, dict[str, Any]], ids: pd.Series) -> pd.Series:
    """True where a model answer exists at least for one field of the matrix."""
    answered: list[bool] = []
    for listing_id in ids:
        record = records.get(str(listing_id)) or {}
        extraction = record.get("extraction")
        block = extraction if isinstance(extraction, dict) else {}
        answered.append(any(value is not None for value in block.values()))
    return pd.Series(answered, index=ids.index, dtype="bool")


def vocabulary_of(spec: dict[str, Any]) -> list[str]:
    """Canonical levels of a group, including the level that collects unknown values.

    A group without an `other` level folds unknown answers into its negative level
    ("газ отсутствует"), which therefore has to be emitted as well; a group with
    `other` keeps `unknown` as a separate level.
    """
    level_names = [level for level, _ in spec["levels"]]
    if spec["catch_all"]:
        level_names.append(str(spec["catch_all"]))
    unknown_level = str(spec["unknown_level"])
    if unknown_level != UNKNOWN_LEVEL:
        level_names.append(unknown_level)
    return list(dict.fromkeys(level_names))


def canonical_levels(values: pd.Series, rules: Sequence[tuple[str, str]],
                     catch_all: str | None) -> tuple[pd.Series, pd.Series]:
    """Map raw values to canonical levels.

    Rules are ordered: the first matching regex wins. Returns the levels (NA means
    "no value matched") and the cleaned text, which is needed to count what fell
    outside the vocabulary.
    """
    cleaned = clean_categorical(values)
    matcher = python_strings(cleaned)
    levels = pd.Series(pd.NA, index=values.index, dtype="string")
    for level, pattern in rules:
        matched = matcher.str.contains(pattern, regex=True, na=False) & levels.isna()
        levels = levels.mask(matched, level)
    if catch_all:
        levels = levels.mask(cleaned.notna() & levels.isna(), catch_all)
    return levels, cleaned


def land_use_tags(raw: pd.DataFrame, index: pd.Index) -> pd.Series:
    """Seller-claimed land-use tags of the card as one normalized string per listing."""
    if "land_use_label" not in raw:
        return pd.Series("", index=index, dtype="string")
    return raw["land_use_label"].map(
        lambda items: "; ".join(str(item).strip() for item in items if str(item).strip())
        if isinstance(items, list) else ""
    )


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


def one_hot_highway(series: pd.Series, builder: FeatureBuilder, source: str,
                    min_count: int) -> list[str]:
    """One-hot the highway by frequency, merging rare levels into `highway_other`."""
    cleaned = series.astype("string").fillna("").str.strip()
    counts = cleaned.value_counts()
    frequent = [level for level, count in counts.items() if count >= min_count and level]
    mapping = {level: f"highway_{index:02d}" for index, level in enumerate(sorted(frequent))}

    created: list[str] = []
    for level, name in mapping.items():
        builder.add_flag(
            name, cleaned.eq(level), BACKBONE_GROUP, "one_hot", source,
            note=f"уровень: {level} ({int(counts[level])} строк)",
        )
        created.append(name)

    other_levels = [level for level in counts.index if level and level not in mapping]
    if other_levels:
        builder.add_flag(
            "highway_other", cleaned.isin(other_levels), BACKBONE_GROUP, "one_hot", source,
            note=f"редкие уровни ({len(other_levels)} шт., "
                 f"{int(counts[other_levels].sum())} строк)",
        )
        created.append("highway_other")
    builder.add_flag("highway_unknown", cleaned.eq(""), BACKBONE_GROUP, "one_hot", source,
                     note="шоссе не указано ни в карточке, ни в извлечении")
    created.append("highway_unknown")
    return created


def build_features(raw: pd.DataFrame, records: dict[str, dict[str, Any]],
                   options: argparse.Namespace) -> tuple[FeatureBuilder, dict[str, Any]]:
    """Build the matrix: the numeric backbone plus the canonical groups."""
    builder = FeatureBuilder()
    index = raw.index
    ids = raw[IDENTIFIER_COLUMN]
    answered = answered_mask(records, ids)

    sources: dict[str, dict[str, Any]] = {}
    ranges: dict[str, dict[str, Any]] = {}

    def llm_field(field: str) -> pd.Series:
        return extraction_values(records, ids, field)

    def ranged(name: str, values: pd.Series) -> pd.Series:
        """Drop implausible values and remember how many were dropped."""
        bounds = VALUE_RANGES.get(name)
        filtered = values.where(values.between(*bounds)) if bounds else values
        ranges[name] = {
            "признак": name,
            "допустимо": f"{bounds[0]:g} … {bounds[1]:g}" if bounds else "—",
            "отброшено": int((values.notna() & filtered.isna()).sum()),
            "минимум LLM": filtered.min(),
            "максимум LLM": filtered.max(),
        }
        return filtered

    # --- числовой каркас: карточка первична, извлечения заполняют пропуски ---------
    structured = {
        "house_area_m2": positive(numeric(raw, "house_area_m2")),
        "land_area_sotka": positive(numeric(raw, "land_area_sotka")),
        "floors": positive(numeric(raw, "floors")),
        "distance_to_mkad_km": positive(numeric(raw, "distance_to_mkad_km")),
        "highway": canonical_highway(raw.get("highway", pd.Series("", index=index))),
    }
    fallback = {
        "house_area_m2": ranged("house_area_m2", positive(to_numeric(
            llm_field("house_area_m2")))),
        "distance_to_mkad_km": ranged("distance_to_mkad_km", positive(to_numeric(
            llm_field("distance_to_mkad_km")))),
        "highway": canonical_highway(llm_field("highway")),
        "land_area_sotka": ranged("land_area_sotka", positive(to_numeric(
            llm_field("land_area_sotka")))),
        "floors": ranged("floors", positive(to_numeric(llm_field("floors")))),
    }
    backbone: dict[str, pd.Series] = {}
    for name in STRUCTURED_FEATURES:
        backbone[name] = fill_gaps(name, BACKBONE_GROUP, structured[name], fallback[name],
                                   sources)
    for name, numerator, denominator in (("house_area_per_floor", "house_area_m2", "floors"),
                                         ("land_to_house_ratio", "land_area_sotka",
                                          "house_area_m2")):
        divisor = backbone[denominator].where(backbone[denominator] > 0)
        backbone[name] = backbone[numerator] / divisor
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
        catch_all = spec["catch_all"]
        if spec.get("card_field"):
            # Метки карточки приоритетны, дальше поля извлечений в указанном порядке.
            card_raw = land_use_tags(raw, index)
            card, card_clean = canonical_levels(card_raw, spec["levels"], catch_all)
            unrecognised = int((card_clean.notna() & card.isna()).sum())
            fallback_levels = pd.Series(pd.NA, index=index, dtype="string")
            for field in spec["llm_fields"]:
                raw_values = llm_field(field)
                values, cleaned = canonical_levels(raw_values, spec["levels"], catch_all)
                unrecognised += int((cleaned.notna() & values.isna()).sum())
                fallback_levels = fallback_levels.where(fallback_levels.notna(), values)
            levels = fill_gaps(prefix, group, card, fallback_levels, sources)
            source = (f"objects.jsonl:{spec['card_field']} | "
                      + " | ".join(f"llm:{field}" for field in spec["llm_fields"]))
            unmapped[prefix] = unrecognised
            # В матрице уровень заполнен всегда: нераспознанный текст уходит в `other`,
            # пустое значение — в `unknown`. Значит, покрытие равно доле строк, по которым
            # модель отвечала или значение дала карточка.
            filled_levels = levels.notna() | answered
            sources[prefix]["from_llm"] = (int(filled_levels.sum())
                                           - int(sources[prefix]["structured"]))
            sources[prefix]["missing"] = int((~filled_levels).sum())
        else:
            raw_values = llm_field(spec["field"])
            levels, cleaned = canonical_levels(raw_values, spec["levels"], catch_all)
            source = f"llm:{spec['field']}"
            unmapped[prefix] = int((cleaned.notna() & levels.isna()).sum())
            # Нераспознанный текст уходит в `other`, отсутствующее — в `unknown` или в
            # отрицательный уровень, поэтому в матрице такая строка заполнена.
            filled_levels = levels.notna() | answered
            sources[prefix] = {
                "group": group,
                "kind": "только LLM",
                "structured": 0,
                "from_llm": int(filled_levels.sum()),
                "missing": int((~filled_levels).sum()),
            }
        levels_by_prefix[prefix] = levels
        specs_by_prefix[prefix] = {**spec, "source": source}
        unknown_level = str(spec["unknown_level"])
        filled = levels.fillna(unknown_level)
        for level in dict.fromkeys([*vocabulary_of(spec), unknown_level]):
            level_rows.append({
                "группа": group,
                "признак": prefix,
                "уровень": level,
                "строк": int((filled.eq(level) & answered).sum()),
            })

    # --- числовые и флаговые признаки групп ---------------------------------------
    number_series: dict[str, pd.Series] = {}
    for group, column, field, kind, note in NUMBER_FEATURES:
        values = to_flag(llm_field(field)) if kind == "flag" else to_numeric(llm_field(field))
        number_series[column] = ranged(column, values).where(answered)
        sources[column] = {
            "group": group, "kind": "только LLM", "structured": 0,
            "from_llm": int(number_series[column].notna().sum()),
            "missing": int(number_series[column].isna().sum()),
        }

    # --- дополнительные строения ---------------------------------------------------
    building_series_map: dict[str, pd.Series] = {}
    for flag_field, area_field, name, _note in BUILDING_SPECS:
        present = to_flag(llm_field(flag_field)).where(answered)
        area = ranged(f"{name}_area_m2",
                      positive(to_numeric(llm_field(area_field)))).where(answered)
        building_series_map[f"{name}_present"] = present
        building_series_map[f"{name}_area_m2"] = area
        for column, values in ((f"{name}_present", present), (f"{name}_area_m2", area)):
            sources[column] = {
                "group": BUILDING_GROUP, "kind": "только LLM", "structured": 0,
                "from_llm": int(values.notna().sum()),
                "missing": int(values.isna().sum()),
            }

    # --- возраст: текущий год минус год постройки ---------------------------------
    year = to_year(llm_field("construction_year"))
    age_from_text = int(year.notna().sum())
    reported_age = to_numeric(llm_field(AGE_COLUMN)).where(
        lambda values: values.between(0, 200))
    age_from_age = int(reported_age.notna().sum())
    year = year.where(year.notna(), (CURRENT_YEAR - reported_age).where(
        reported_age.notna()))
    age = (CURRENT_YEAR - year).where(lambda values: values >= 0)
    age = ranged(AGE_COLUMN, age).where(answered)
    number_series[AGE_COLUMN] = age
    sources[AGE_COLUMN] = {
        "group": "Возраст", "kind": "только LLM", "structured": 0,
        "from_llm": int(age.notna().sum()), "missing": int(age.isna().sum()),
    }

    # --- вывод колонок в фиксированном порядке ------------------------------------
    for name in BACKBONE_ORDER:
        kind, source, note = BACKBONE_META[name]
        if name == "highway":
            one_hot_highway(backbone[name], builder, source, options.one_hot_min_count)
        else:
            builder.add(name, backbone[name], BACKBONE_GROUP, kind, source, note)

    group_columns: dict[str, pd.Series] = {**number_series, **building_series_map}
    number_meta = {column: (group, kind, note)
                   for group, column, _field, kind, note in NUMBER_FEATURES}
    building_notes = {
        **{f"{name}_present": f"в извлечении есть {note} ({flag_field})"
           for flag_field, _area, name, note in BUILDING_SPECS},
        **{f"{name}_area_m2": f"площадь: {note}"
           for _flag, _area, name, note in BUILDING_SPECS},
    }
    for name in EMIT_ORDER:
        if name in specs_by_prefix:
            spec = specs_by_prefix[name]
            levels = levels_by_prefix[name]
            vocabulary = vocabulary_of(spec)
            unknown_level = str(spec["unknown_level"])
            # Группы без уровня «other» схлопывают неизвестное в свой отрицательный
            # уровень (например, газ отсутствует), поэтому он и получает строки с NA.
            folded = unknown_level != UNKNOWN_LEVEL
            effective = levels.fillna(unknown_level) if folded else levels
            for level in vocabulary:
                builder.add_flag(
                    f"{name}_{level}", effective.eq(level), spec["group"], "level",
                    spec["source"], note=f"{spec['note']}; уровень: {level}",
                    answered=answered,
                )
            if not folded:
                builder.add_flag(
                    f"{name}_{UNKNOWN_LEVEL}", levels.isna(), spec["group"], "level",
                    spec["source"], note=f"{spec['note']}; уровень: {UNKNOWN_LEVEL}",
                    answered=answered,
                )
            continue
        if name in group_columns:
            group, kind, note = number_meta.get(
                name, (BUILDING_GROUP, "flag" if name.endswith("_present") else "count",
                       building_notes.get(name, "")))
            if name == AGE_COLUMN:
                group, kind, note = ("Возраст", "derived", "текущий год минус год постройки")
            builder.add(name, group_columns[name], group, kind, f"llm:{name}", note)
            continue
        raise AssertionError(f"Признак {name} не описан.")

    target = numeric(raw, TARGET_COLUMN)
    builder.add(TARGET_COLUMN, target, "target", "target", TARGET_COLUMN,
                "цена предложения, руб. — не цена сделки")
    builder.add(LOG_TARGET_COLUMN, np.log(target.where(target > 0)), "target", "target",
                TARGET_COLUMN, "логарифм цены: вариант цели для скошенного распределения")

    extraction_fields = list(dict.fromkeys(
        [field for spec in CATEGORICAL_GROUPS
         for field in ([spec["field"]] if spec.get("field") else list(spec["llm_fields"]))]
        + [field for _group, _column, field, _kind, _note in NUMBER_FEATURES]
        + [flag for flag, _area, _name, _note in BUILDING_SPECS]
        + [area for _flag, area, _name, _note in BUILDING_SPECS]
        + ["construction_year", AGE_COLUMN]
    ))
    column_by_field = {
        "bedrooms_count": "bedrooms", "bathrooms_count": "bathrooms",
        "has_attic": "attic_present", "has_basement": "basement_present",
        "construction_year": AGE_COLUMN,
        **{flag: f"{name}_present" for flag, _area, name, _note in BUILDING_SPECS},
        **{area: f"{name}_area_m2" for _flag, area, name, _note in BUILDING_SPECS},
        **{field: field for field in BACKBONE_LLM_FIELDS.values()},
    }
    coverage_column = dict(zip(
        [field for _group, column, field, _kind, _note in NUMBER_FEATURES],
        [column for _group, column, _field, _kind, _note in NUMBER_FEATURES],
    ))
    column_by_field.update(coverage_column)
    extraction_summary = pd.DataFrame([
        {
            "поле": field,
            "признак": column_by_field.get(field, field),
            "заполнено": int(extraction_values(records, ids, field).notna().sum()),
            "пусто": int(extraction_values(records, ids, field).isna().sum()),
        }
        for field in extraction_fields
    ])
    extraction_summary["доля, %"] = (
        extraction_summary["заполнено"] / max(len(ids), 1) * 100
    ).round(1)

    audit = {
        "sources": sources,
        "disagreements": disagreements,
        "levels": pd.DataFrame(level_rows),
        "unmapped": unmapped,
        "ranges": pd.DataFrame(list(ranges.values())),
        "extraction_summary": extraction_summary,
        "answered": int(answered.sum()),
        "age_from_text": age_from_text,
        "age_from_age": age_from_age,
    }
    return builder, audit


# --------------------------------------------------------------------------- #
# Report
# --------------------------------------------------------------------------- #

def coverage_table(features: pd.DataFrame, dictionary: pd.DataFrame) -> pd.DataFrame:
    """Coverage and basic statistics for every feature column."""
    rows: list[dict[str, Any]] = []
    for column in features.columns:
        values = features[column]
        filled = values.notna()
        rows.append({
            "column": column,
            "non_null": int(filled.sum()),
            "coverage_pct": round(float(filled.mean()) * 100, 1),
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
            filled / row_count * 100 if row_count else 0.0
        )
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


def markdown_safe_table(frame: pd.DataFrame, max_rows: int = 30) -> str:
    """Markdown table for the feature report, truncated to the first `max_rows` rows.

    Numbers are rendered the way the other feature reports do it: thousands separated by
    a space, fractions with one decimal, missing values as a dash. Percentage and coverage
    columns therefore read as "97,6", not as "97,600".
    """
    if frame.empty:
        return "_Нет данных._"
    shown = frame.head(max_rows)
    headers = [str(name).replace("|", "\\|") for name in shown.columns]
    lines = ["| " + " | ".join(headers) + " |",
             "| " + " | ".join("---" for _ in headers) + " |"]
    for row in shown.itertuples(index=False, name=None):
        lines.append("| " + " | ".join(display_value(value) for value in row) + " |")
    if len(frame) > max_rows:
        lines.append(f"\nПоказаны первые {max_rows} из {len(frame)} строк.")
    return "\n".join(lines)


def display_value(value: Any) -> str:
    """One table cell: missing values as a dash, integers grouped, floats with one digit."""
    if value is None or (not isinstance(value, str) and pd.isna(value)):
        return "—"
    if isinstance(value, str):
        return value.replace("|", "\\|")
    if isinstance(value, (int, np.integer)):
        return counted(int(value))
    number = float(value)
    if number.is_integer():
        return counted(int(number)) if abs(number) >= 1000 else str(int(number))
    return f"{number:.1f}".replace(".", ",")


def write_report(report_dir: Path, input_path: Path, features_path: Path, cache_path: Path,
                 matrix: pd.DataFrame, feature_columns: list[str], audit: dict[str, Any],
                 constants: list[str], options: argparse.Namespace, invalid_lines: int,
                 dropped_target: int, dictionary: pd.DataFrame,
                 cache_stats: dict[str, Any], prompts: dict[str, str]) -> None:
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
    highway_levels = dictionary.loc[dictionary["kind"].eq("one_hot"), ["column", "note"]]
    level_notes = pd.DataFrame([
        {
            "признак": spec["prefix"],
            "уровни": ", ".join(vocabulary_of(spec)),
            "неизвестное": spec["unknown_level"],
            "вне словаря": audit["unmapped"][spec["prefix"]],
        }
        for spec in CATEGORICAL_GROUPS
    ])

    features_total = len(feature_columns)
    # Column families come from the dictionary, not from the values: a column that is
    # empty in this sample is still a 0/1 indicator, not a numeric feature.
    kind_by_column = dict(zip(dictionary["column"], dictionary["kind"]))
    indicator_kinds = {"level", "flag", "one_hot"}
    binary = [column for column in feature_columns
              if kind_by_column.get(column) in indicator_kinds]
    prompts_text = ", ".join(f"`{name}` (sha `{digest[:8]}`)"
                             for name, digest in prompts.items()) or "нет"
    model_text = (f"`{options.model}` на `{options.host}`"
                  if options.llm_fill else "не использовалась (`--no-llm`)")

    report = [
        "# Компактная матрица v2: числовой каркас и группы признаков",
        "",
        f"Источник: `{input_path}`",
        f"Матрица: `{features_path}`",
        f"Кэш ответов модели: `{cache_path}`",
        f"Промпты: {prompts_text}",
        f"Модель: {model_text}; версия промпта `{PROMPT_VERSION}`.",
        f"Строк: **{counted(row_count)}**; колонок всего: **{matrix.shape[1]}** "
        f"(признаков **{features_total}**, плюс `{IDENTIFIER_COLUMN}`, цель "
        f"`{TARGET_COLUMN}`, вариант цели `{LOG_TARGET_COLUMN}` и ключ групп "
        f"`{GROUP_COLUMN}`).",
        f"Признаки: **{features_total - len(binary)}** числовых и **{len(binary)}** бинарных "
        "(категориальные уровни развёрнуты в колонки-индикаторы).",
        f"Битых строк JSON: **{invalid_lines}**; строк без положительной цены отброшено: "
        f"**{dropped_target}**.",
        "",
        "## Извлечения локальной модели",
        "Извлечения запрашиваются у локального сервера Ollama отдельными промптами для "
        "категориальных и для числовых признаков; карточка объявления передаётся как "
        "контекст, а цена и метки платформы — нет: они связаны с целевой переменной, "
        "и признаки, собранные из них, были бы утечкой.",
        f"Ответов в кэше: **{counted(len(cache_stats.get('records') or {}))}**; "
        f"взято из кэша без запроса: **{counted(cache_stats.get('from_cache', 0))}**; "
        f"запрошено в этом запуске: **{counted(cache_stats.get('requested', 0))}**; "
        f"объявлений с ошибкой ответа: **{counted(cache_stats.get('failed', 0))}**; "
        f"время запросов: **{cache_stats.get('seconds', 0.0) / 60:.1f} мин**.",
        f"Объявлений с хотя бы одним ответом модели: **{counted(audit['answered'])}** "
        f"из **{counted(row_count)}**; у остальных признаки групп остаются NaN, а не нулём.",
        "",
        "### Заполненность полей извлечения",
        markdown_safe_table(audit["extraction_summary"]),
        "",
        "## Группы признаков",
        "Числовой каркас берётся из карточки объявления (извлечения заполняют пропуски), "
        "остальные группы собираются из ответов модели. У каждой группы свой словарь "
        "уровней, поэтому синонимы не разъезжаются по разным колонкам.",
        markdown_safe_table(groups),
        "",
        "## Признаки и источники значений",
        "Источник «карточка + LLM» означает, что значение карточки приоритетно и только "
        "пустые места заполнены извлечением.",
        markdown_safe_table(families, max_rows=40),
        "",
        "## Канонические уровни групп",
        "Уровни описаны в скрипте (`CATEGORICAL_GROUPS`) и выводятся всегда, даже если "
        "уровень пуст: схема матрицы не зависит от выборки. Текст вне словаря попадает в "
        "уровень `other` (если он есть у группы) либо в тот уровень, куда словарь относит "
        f"неизвестное значение; всего таких значений: **{counted(unmapped_total)}**.",
        markdown_safe_table(level_notes),
        "",
        markdown_safe_table(levels, max_rows=60),
        "",
        "## Уровни шоссе (one-hot по частоте)",
        "Шоссе — часть числового каркаса: уровень получает отдельную колонку, если "
        f"встречается не реже {options.one_hot_min_count} раз; редкие уровни объединены "
        "в `highway_other`, отсутствие значения — в `highway_unknown`. Значения карточки и "
        "извлечения приводятся к одному виду: слово «шоссе» отбрасывается.",
        markdown_safe_table(highway_levels, max_rows=40),
        "",
        "## Проверка диапазонов",
        "Значения вне осмысленного диапазона заменены на NaN: это ошибки модели, а не "
        f"измерения. Возраст дома считается как {CURRENT_YEAR} минус год постройки; год "
        f"взят из поля `construction_year` (**{audit['age_from_text']}** строк), а если "
        f"модель вернула только возраст — из `house_age` (**{audit['age_from_age']}** строк).",
        markdown_safe_table(audit["ranges"]),
        "",
        "## Расхождения карточки и извлечений",
        "Строки, где оба источника дали значение, но значения разошлись (для чисел — "
        "относительное расхождение больше 1 %). В матрицу попадает значение карточки.",
        markdown_safe_table(disagreements),
        "",
        "## Особенности матрицы",
        "- Пропуски **не заполняются**: NaN остаётся, потому что "
        "`HistGradientBoostingRegressor`, XGBoost и LightGBM обрабатывают NaN сами, "
        "а заполнение до разбиения на фолды переносит информацию из теста в обучение. "
        "Для линейных моделей заполняйте NaN медианой, посчитанной **только на "
        "обучающей части** каждого фолда.",
        "- Группы газ, отопление, вода, канализация и отделка не имеют уровня «неизвестно»: "
        "как и требует словарь, неупомянутый или нераспознанный ответ попадает в "
        "отрицательный уровень (`otsutstvuet`, `without_repair`). Строки, по которым "
        "модель вообще не отвечала, остаются NaN — «не спрашивали» не становится «нет».",
        "- Планировка и дополнительные строения идут и флагом (есть/нет), и числом: "
        "0 — модель явно не нашла признак в тексте, NaN — ответа по объявлению нет.",
        "- Возраст дома — единственная колонка группы «Возраст»: год постройки "
        f"восстанавливается как `{CURRENT_YEAR} - {AGE_COLUMN}`, отдельная колонка была бы "
        "линейно зависимой копией.",
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
        f"- Кэш ответов модели: `{cache_path}`",
        f"- Словарь признаков: `{report_dir / 'feature_dictionary.csv'}`",
        f"- Покрытие и статистика: `{report_dir / 'feature_coverage.csv'}`",
        f"- Сводка по полям извлечения: `{report_dir / 'extraction_summary.csv'}`",
        f"- Уровни групп: `{report_dir / 'feature_levels.csv'}`",
        f"- Проверка диапазонов: `{report_dir / 'value_ranges.csv'}`",
        f"- Расхождения источников: `{report_dir / 'disagreements.csv'}`",
        f"- Отчёт: `{report_dir / 'report.md'}`",
    ]
    (report_dir / "report.md").write_text("\n".join(report) + "\n", encoding="utf-8")


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #

def run(input_path: Path, cache_path: Path, features_path: Path, report_dir: Path,
        options: argparse.Namespace) -> None:
    raw, invalid_lines = read_jsonl(input_path)
    target = numeric(raw, TARGET_COLUMN)
    keep = target.gt(0)
    dropped_target = int((~keep).sum())
    raw = raw.loc[keep].reset_index(drop=True)
    if raw.empty:
        raise SystemExit("Не осталось строк с положительной ценой.")
    if IDENTIFIER_COLUMN not in raw:
        raise SystemExit(f"В {input_path} нет колонки {IDENTIFIER_COLUMN}.")
    if options.limit and options.limit < len(raw):
        LOG.warning("Ограничение --limit %s: из %s объявлений обрабатываются первые %s",
                    options.limit, len(raw), options.limit)
        raw = raw.head(options.limit).reset_index(drop=True)

    records, cache_stats = collect_extractions(raw, cache_path, options)
    prompts: dict[str, str] = {}
    if options.llm_fill:
        prompts = {
            "prompt_v2_cat.txt": prompt_hash(load_prompt(options.prompt_cat)),
            "prompt_v2_num.txt": prompt_hash(load_prompt(options.prompt_num)),
        }

    builder, audit = build_features(raw, records, options)
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

    matrix = pd.concat([raw[IDENTIFIER_COLUMN].reset_index(drop=True), features], axis=1)

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
    audit["extraction_summary"].to_csv(report_dir / "extraction_summary.csv", index=False,
                                       encoding="utf-8-sig")
    audit["levels"].to_csv(report_dir / "feature_levels.csv", index=False,
                           encoding="utf-8-sig")
    audit["ranges"].to_csv(report_dir / "value_ranges.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame([{"признак": name, "расхождений": count}
                  for name, count in audit["disagreements"].items()]).to_csv(
        report_dir / "disagreements.csv", index=False, encoding="utf-8-sig")

    feature_columns = [column for column in features.columns
                       if column not in NON_FEATURE_COLUMNS]
    write_report(report_dir, input_path, features_path, cache_path, matrix, feature_columns,
                 audit, constants, options, invalid_lines, dropped_target, dictionary,
                 cache_stats, prompts)

    filled_backbone = sum(stats["from_llm"] for name, stats in audit["sources"].items()
                          if name in STRUCTURED_FEATURES)
    answered_rows = audit["answered"]
    say("")
    say(f"Матрица: строк {counted(len(matrix))}, колонок {matrix.shape[1]} "
        f"(признаков {counted(len(feature_columns))})")
    say(f"Заполнено из извлечений (каркас): {counted(filled_backbone)} значений; "
        f"ответов модели по объявлениям: {counted(answered_rows)} из "
        f"{counted(len(matrix))}")
    say("Группы признаков:")
    groups = group_table(dictionary, audit, len(matrix))
    for name, columns_count, coverage in zip(groups["группа"], groups["колонок"],
                                            groups["среднее покрытие, %"]):
        say(f"  {name:<26} колонок {int(columns_count):>2}, покрытие {coverage} %")
    say(f"Матрица признаков: {features_path}")
    say(f"Кэш извлечений: {cache_path}")
    say(f"Отчёт и словарь: {report_dir}")
    if options.llm_fill and cache_stats.get("missing"):
        missing = int(cache_stats["missing"])
        say(f"Без ответа модели осталось {counted(missing)} "
            f"{plural(missing, 'объявление', 'объявления', 'объявлений')} — их признаки "
            f"групп в матрице пустые. Перезапустите ту же команду: готовые ответы "
            f"читаются из кэша и не запрашиваются повторно.")
    elif not options.llm_fill:
        say("Извлечения были отключены (--no-llm): полный прогон — та же команда без этого "
            "ключа, кэш заполнится заново.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Построение компактной матрицы v2: числовой каркас из карточки, "
                    "пропуски и группы признаков — извлечениями локальной модели Ollama."
    )
    parser.add_argument(
        "--input", type=Path, default=default_input_path(),
        help="Путь к objects.jsonl (по умолчанию data/dataset/objects.jsonl).",
    )
    parser.add_argument(
        "--prompt-cat", type=Path, default=default_prompt_cat_path(),
        help="Промпт для категориальных признаков (по умолчанию prompts/prompt_v2_cat.txt).",
    )
    parser.add_argument(
        "--prompt-num", type=Path, default=default_prompt_num_path(),
        help="Промпт для числовых признаков (по умолчанию prompts/prompt_v2_num.txt).",
    )
    parser.add_argument(
        "--cache", type=Path, default=default_cache_path(),
        help="Кэш ответов модели (по умолчанию data/features/llm_v2_extractions.jsonl).",
    )
    parser.add_argument(
        "--features-file", type=Path, default=default_features_path(),
        help="Куда записать матрицу (по умолчанию data/features/features_v2.csv).",
    )
    parser.add_argument(
        "--report-dir", type=Path, default=default_report_dir(),
        help="Каталог для отчёта и словаря (по умолчанию reports/features_v2).",
    )
    parser.add_argument(
        "--model", default=DEFAULT_MODEL,
        help=f"Имя модели в Ollama (по умолчанию {DEFAULT_MODEL}).",
    )
    parser.add_argument(
        "--host", default=DEFAULT_HOST,
        help=f"Адрес сервера Ollama (по умолчанию {DEFAULT_HOST}).",
    )
    parser.add_argument(
        "--num-ctx", type=int, default=8192,
        help="Размер контекста модели в токенах (по умолчанию 8192).",
    )
    parser.add_argument(
        "--think", action="store_true",
        help="Включить рассуждения модели: точнее, но в разы медленнее.",
    )
    parser.add_argument(
        "--retries", type=int, default=1,
        help="Сколько раз повторять запрос при ошибке (по умолчанию 1).",
    )
    parser.add_argument(
        "--parallel", type=int, default=1,
        help="Сколько объявлений запрашивать одновременно (по умолчанию 1).",
    )
    parser.add_argument(
        "--limit", type=int, default=0,
        help="Обработать только первые N объявлений — для быстрой проверки.",
    )
    parser.add_argument(
        "--refresh", action="store_true",
        help="Игнорировать кэш и перезапросить все объявления.",
    )
    parser.add_argument(
        "--log-every", type=int, default=10,
        help="Писать строку прогресса каждые N объявлений (по умолчанию 10; "
             "0 — только итоговая сводка).",
    )
    parser.add_argument(
        "--one-hot-min-count", type=int, default=10,
        help="Минимальное число строк, чтобы уровень шоссе получил свою колонку.",
    )
    parser.add_argument(
        "--no-llm", dest="llm_fill", action="store_false", default=True,
        help="Не обращаться к Ollama: собрать матрицу только из полей карточки.",
    )
    parser.add_argument("--verbose", action="store_true",
                        help="Писать строку по каждому объявлению и подробные сообщения.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(levelname)s: %(message)s",
    )
    run(
        args.input.expanduser(),
        args.cache.expanduser(),
        args.features_file.expanduser(),
        args.report_dir.expanduser(),
        args,
    )


if __name__ == "__main__":
    main()
