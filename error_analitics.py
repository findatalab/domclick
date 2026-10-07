#!/usr/bin/env python3
"""Разбор топ-N ошибок предсказаний одного запуска обучения.

Вход:  `experiments/<NN>/predictions.csv` — out-of-fold прогнозы обучающих скриптов
       (факт, прогноз, остаток и прогноз базовой модели).
Выход: `experiments/<NN>/errors/` — `top_errors.csv`, `error_summary.csv`,
       `figures/top_errors.png` и отчёт `report.md`.

Относительная ошибка считается от факта (цены предложения):

    rel_error = (прогноз - факт) / факт

Ранжирование — по модулю `|rel_error|`: положительное значение означает переоценку,
отрицательное — недооценку. Матрица признаков из `meta.json` подтягивается только для
колонок-пояснений (площадь, расстояние, возраст); сами прогнозы не пересчитываются.
Скрипт не меняет запуск: он пишет только в подкаталог `errors/`, поэтому карточка
`meta.json` и реестр `index.csv` остаются прежними.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import report_utils


LOG = logging.getLogger("domclick.errors")

#: Колонки, без которых разбор ошибок невозможен.
REQUIRED_COLUMNS = ("listing_id", "actual_rub", "predicted_rub")

#: Метрика ранжирования: модуль относительной ошибки по факту.
RANK_METRIC = "abs_rel_error"

#: Понятные колонки матрицы признаков, которые помогают читать крупные ошибки.
CONTEXT_COLUMNS = ("house_area_m2", "land_area_sotka", "floors", "distance_to_mkad_km",
                   "listing_age_days")
CONTEXT_LABELS = {
    "house_area_m2": "площадь дома, м²",
    "land_area_sotka": "участок, сот.",
    "floors": "этажей",
    "distance_to_mkad_km": "до МКАД, км",
    "listing_age_days": "возраст, дн.",
}

#: Сводные метрики: ключ, подпись в отчёте и число знаков после запятой.
SUMMARY_ROWS = (
    ("rows", "Объявлений в выборке", 0),
    ("top_n", "Объявлений в топе", 0),
    ("mape_pct", "MAPE по выборке, %", 1),
    ("median_abs_rel_error_pct", "Медиана модуля отн. ошибки, %", 1),
    ("p90_abs_rel_error_pct", "90-й процентиль модуля отн. ошибки, %", 1),
    ("mean_rel_error_pct", "Средняя отн. ошибка (смещение), %", 1),
    ("top_n_share_of_abs_error_pct", "Доля топа в абсолютной ошибке, %", 1),
    ("top_n_overpredicted", "Переоценок в топе", 0),
    ("top_n_underpredicted", "Недооценок в топе", 0),
    ("abs_error_rub_total", "Абсолютная ошибка всей выборки, ₽", 0),
    ("abs_error_rub_top_n", "Абсолютная ошибка топа, ₽", 0),
    ("baseline_mape_pct", "MAPE базовой модели (среднее), %", 1),
)

QUANTILES = (0.0, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99, 1.0)


def _number(value: Any, digits: int = 1) -> str:
    """Число для отчёта: пробел в тысячах, запятая в дробях, «—» вместо NaN."""
    if value is None or (not isinstance(value, str) and pd.isna(value)):
        return "—"
    return f"{float(value):,.{digits}f}".replace(",", " ").replace(".", ",")


def latest_run_dir(experiments_dir: Path) -> Path:
    """Самый новый по номеру запуск, в котором уже есть прогнозы."""
    if not experiments_dir.is_dir():
        raise SystemExit(f"Нет каталога экспериментов: {experiments_dir}")
    numbers = sorted(
        (int(path.name) for path in experiments_dir.iterdir()
         if path.is_dir() and path.name.isdigit()),
        reverse=True,
    )
    for number in numbers:
        candidate = experiments_dir / f"{number:02d}"
        if (candidate / "predictions.csv").is_file():
            return candidate
    raise SystemExit(
        f"В {experiments_dir} нет запусков с predictions.csv; укажите --run-dir."
    )


def resolve_run_dir(value: str | None, experiments_dir: Path) -> Path:
    """Каталог запуска по пути или номеру; по умолчанию — последний по номеру."""
    if not value:
        candidate = latest_run_dir(experiments_dir)
    else:
        candidate = Path(value).expanduser()
        if not candidate.is_dir():
            if not candidate.name.isdigit():
                raise SystemExit(f"Не найден каталог запуска: {candidate}")
            candidate = experiments_dir / candidate.name
    if not (candidate / "predictions.csv").is_file():
        raise SystemExit(
            f"В {candidate} нет predictions.csv — сначала обучите модель (train_*.py)."
        )
    return candidate.resolve()


def load_predictions(path: Path) -> tuple[pd.DataFrame, int]:
    """Прогнозы запуска; строки без факта, прогноза или с ценой <= 0 отбрасываются."""
    try:
        frame = pd.read_csv(path)
    except FileNotFoundError as exc:
        raise SystemExit(f"Не найден файл прогнозов: {path}") from exc
    except OSError as exc:
        raise SystemExit(f"Не удалось прочитать {path}: {exc}") from exc

    missing = [column for column in REQUIRED_COLUMNS if column not in frame.columns]
    if missing:
        raise SystemExit(f"В {path} нет колонок: {', '.join(missing)}.")

    for column in ("actual_rub", "predicted_rub"):
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
        frame[column] = frame[column].replace([np.inf, -np.inf], np.nan)

    unusable = (frame["actual_rub"].isna() | frame["predicted_rub"].isna()
                | frame["actual_rub"].le(0))
    dropped = int(unusable.sum())
    if dropped:
        LOG.warning("Строк без пригодного факта или прогноза отброшено: %s", dropped)
        frame = frame.loc[~unusable]
    if frame.empty:
        raise SystemExit(f"В {path} не осталось строк с фактом и прогнозом.")

    frame = frame.reset_index(drop=True)
    frame["listing_id"] = _as_identifier(frame["listing_id"])
    return frame, dropped


def _as_identifier(series: pd.Series) -> pd.Series:
    """Целочисленные id без дробной части; нечисловые оставляем как есть."""
    if pd.api.types.is_numeric_dtype(series) and series.notna().all():
        return series.astype("int64")
    return series


def add_error_columns(frame: pd.DataFrame) -> pd.DataFrame:
    """Относительная ошибка и её модуль; положительный знак — переоценка."""
    result = frame.copy()
    result["rel_error"] = (result["predicted_rub"] - result["actual_rub"]) / result["actual_rub"]
    result[RANK_METRIC] = result["rel_error"].abs()
    if "residual_rub" not in result:
        result["residual_rub"] = result["actual_rub"] - result["predicted_rub"]
    if "abs_error_rub" not in result:
        result["abs_error_rub"] = result["residual_rub"].abs()
    result["direction"] = np.where(result["rel_error"].ge(0), "переоценка", "недооценка")
    return result


def load_context(run_dir: Path) -> tuple[dict[str, Any], pd.DataFrame, str]:
    """Карточка запуска и колонки-пояснения из матрицы признаков.

    Третий элемент — причина, по которой контекст не добавлен (пустая строка при успехе).
    """
    try:
        manifest = json.loads((run_dir / "meta.json").read_text(encoding="utf-8"))
    except FileNotFoundError:
        return {}, pd.DataFrame(), "карточка meta.json не найдена"
    except (OSError, ValueError) as exc:
        return {}, pd.DataFrame(), f"карточка meta.json не прочитана ({exc})"

    input_path = Path(str(manifest.get("data", {}).get("input", "")))
    if not str(input_path):
        return manifest, pd.DataFrame(), "в карточке не указан входной файл"
    if not input_path.is_absolute():
        input_path = report_utils.REPO_ROOT / input_path
    if not input_path.is_file():
        return manifest, pd.DataFrame(), f"матрица признаков не найдена ({input_path})"

    try:
        matrix = pd.read_csv(input_path)
    except (OSError, ValueError) as exc:
        return manifest, pd.DataFrame(), f"матрица признаков не прочитана ({exc})"

    columns = [column for column in CONTEXT_COLUMNS if column in matrix.columns]
    if not columns or "listing_id" not in matrix.columns:
        return manifest, pd.DataFrame(), "в матрице признаков нет колонок-пояснений"
    context = matrix[["listing_id", *columns]].copy()
    context["listing_id"] = _as_identifier(context["listing_id"])
    return manifest, context, ""


def summarize(frame: pd.DataFrame, top: pd.DataFrame) -> pd.DataFrame:
    """Числа, описывающие ошибки выборки и вклад топа; длинный формат `metric,value`."""
    total_abs = float(frame["abs_error_rub"].sum())
    top_abs = float(top["abs_error_rub"].sum())
    rows: list[tuple[str, float]] = [
        ("rows", float(len(frame))),
        ("top_n", float(len(top))),
        ("mape_pct", float(frame[RANK_METRIC].mean() * 100)),
        ("median_abs_rel_error_pct", float(frame[RANK_METRIC].median() * 100)),
        ("p90_abs_rel_error_pct", float(frame[RANK_METRIC].quantile(0.9) * 100)),
        ("mean_rel_error_pct", float(frame["rel_error"].mean() * 100)),
        ("top_n_share_of_abs_error_pct",
         float(top_abs / total_abs * 100) if total_abs else np.nan),
        ("top_n_overpredicted", float(int(top["rel_error"].ge(0).sum()))),
        ("top_n_underpredicted", float(int(top["rel_error"].lt(0).sum()))),
        ("abs_error_rub_total", total_abs),
        ("abs_error_rub_top_n", top_abs),
    ]
    if "baseline_rub" in frame.columns:
        baseline = pd.to_numeric(frame["baseline_rub"], errors="coerce")
        usable = frame["actual_rub"].gt(0) & baseline.notna()
        if usable.any():
            baseline_rel = ((baseline - frame["actual_rub"]).abs() / frame["actual_rub"])[usable]
            rows.append(("baseline_mape_pct", float(baseline_rel.mean() * 100)))
    return pd.DataFrame(rows, columns=["metric", "value"])


def top_display(top: pd.DataFrame, context_columns: list[str]) -> pd.DataFrame:
    """Таблица топа для отчёта: деньги и проценты уже отформатированы."""
    records = top.to_dict("records")
    columns: dict[str, list[str]] = {
        "#": [str(rank) for rank in range(1, len(records) + 1)],
        "listing_id": [str(record["listing_id"]) for record in records],
        "факт, ₽": [_number(record["actual_rub"], 0) for record in records],
        "прогноз, ₽": [_number(record["predicted_rub"], 0) for record in records],
        "отн. ошибка, %": [_number(record[RANK_METRIC] * 100, 1) for record in records],
        "направление": [str(record["direction"]) for record in records],
    }
    for column in context_columns:
        columns[CONTEXT_LABELS[column]] = [_number(record.get(column), 1) for record in records]
    return pd.DataFrame(columns)


def summary_display(summary: pd.DataFrame) -> pd.DataFrame:
    """Сводка для отчёта: подписи и разрядность берутся из `SUMMARY_ROWS`."""
    values = dict(zip(summary["metric"], summary["value"]))
    return pd.DataFrame([
        {"Показатель": label, "Значение": _number(values[name], digits)}
        for name, label, digits in SUMMARY_ROWS if name in values
    ])


def quantile_display(frame: pd.DataFrame) -> pd.DataFrame:
    """Квантили модуля относительной ошибки в процентах."""
    values = frame[RANK_METRIC].quantile(list(QUANTILES))
    return pd.DataFrame({
        "Квантиль": [f"{quantile * 100:.0f} %" for quantile in values.index],
        "Модуль отн. ошибки, %": [_number(value * 100, 1) for value in values],
    })


def observations(frame: pd.DataFrame, top: pd.DataFrame, summary: pd.DataFrame) -> list[str]:
    """Короткие выводы: вклад топа, направление ошибок и ориентиры по выборке."""
    values = dict(zip(summary["metric"], summary["value"]))
    worst = top.iloc[0]
    lines = [
        f"Топ-{len(top)} даёт **{_number(values['top_n_share_of_abs_error_pct'], 1)} %** всей "
        f"абсолютной ошибки выборки ({_number(values['abs_error_rub_top_n'], 0)} ₽ из "
        f"{_number(values['abs_error_rub_total'], 0)} ₽).",
        f"Знак ошибки в топе: переоценка — {_number(values['top_n_overpredicted'], 0)}, "
        f"недооценка — {_number(values['top_n_underpredicted'], 0)}.",
        f"По всей выборке MAPE = {_number(values['mape_pct'], 1)} %, медиана |отн. ошибки| = "
        f"{_number(values['median_abs_rel_error_pct'], 1)} %, смещение "
        f"{_number(values['mean_rel_error_pct'], 1)} % (плюс — систематическая переоценка).",
    ]
    if "baseline_mape_pct" in values:
        lines.append(
            f"Базовая модель (среднее по обучающей части) даёт MAPE "
            f"{_number(values['baseline_mape_pct'], 1)} %."
        )
    lines.append(
        f"Худший объект: `listing_id={worst['listing_id']}` — факт "
        f"{_number(worst['actual_rub'], 0)} ₽, прогноз {_number(worst['predicted_rub'], 0)} ₽, "
        f"отн. ошибка {_number(worst[RANK_METRIC] * 100, 1)} %."
    )
    return lines


def plot_top_errors(top: pd.DataFrame, path: Path) -> Path:
    """Столбцы модуля относительной ошибки топа: сверху — самая большая ошибка."""
    path.parent.mkdir(parents=True, exist_ok=True)
    records = top.to_dict("records")
    labels = [
        f"{record['listing_id']} (факт {_number(record['actual_rub'], 0)} ₽)"
        for record in records
    ]
    values = (top[RANK_METRIC] * 100).to_numpy(dtype=float)
    figure, axes = plt.subplots(figsize=(9.0, max(3.0, 0.45 * len(records) + 1.4)))
    axes.barh(labels[::-1], values[::-1], color="#c0392b")
    axes.set_xlabel("Относительная ошибка, %")
    axes.set_title(f"Топ-{len(records)} объявлений по относительной ошибке")
    axes.grid(axis="x", alpha=0.3)
    axes.margins(x=0.14)
    for position, value in enumerate(values[::-1]):
        axes.text(value, position, f" {_number(value, 1)} %", va="center", fontsize=8)
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)
    return path


def build_report(*, run_dir: Path, predictions_path: Path, output_dir: Path,
                 manifest: dict[str, Any], frame: pd.DataFrame, top: pd.DataFrame,
                 summary: pd.DataFrame, context_columns: list[str], context_note: str,
                 dropped: int) -> str:
    """Отчёт: контекст запуска, определение метрики, топ, сводка, наблюдения, артефакты."""
    data = manifest.get("data", {})
    cv = manifest.get("cv", {})
    if manifest:
        context_lines = [
            f"- Запуск **{manifest.get('run_id', run_dir.name)}**: скрипт "
            f"`{manifest.get('script', '—')}`, модель `{manifest.get('model', '—')}`.",
            f"- Данные: `{data.get('input', '—')}`, строк: **{_number(data.get('rows'), 0)}**, "
            f"признаков: **{_number(data.get('features'), 0)}**.",
            f"- Валидация: `{cv.get('scheme', '—')}`, фолдов: {_number(cv.get('folds'), 0)}, "
            f"seed {_number(cv.get('seed'), 0)}; метрика-ориентир: "
            f"`{manifest.get('primary_metric', '—')}` = "
            f"{_number(manifest.get('primary_value'), 0)}.",
        ]
    else:
        context_lines = ["- Карточка `meta.json` недоступна, контекст запуска не восстановлен."]

    lines = [
        f"# Топ-{len(top)} объявлений по относительной ошибке — запуск {run_dir.name}",
        "",
        "## Контекст запуска",
        *context_lines,
        f"- Прогнозы: `{report_utils.display_path(predictions_path)}` "
        f"(out-of-fold: каждый объект предсказан моделью, не видевшей его фолд).",
        "",
        "## Как считается относительная ошибка",
        "Ранжирование — по модулю относительной ошибки прогноза к факту:",
        "",
        "```",
        "rel_error = (прогноз - факт) / факт;    ранжирование по |rel_error|",
        "```",
        "",
        f"Факт — `{data.get('target', 'asking_price_rub')}` (цена предложения). Положительный "
        "`rel_error` — переоценка, отрицательный — недооценка. Ошибка нормируется на факт, "
        "поэтому у дешёвых объектов она крупнее при той же абсолютной ошибке и в топе они "
        "встречаются чаще; абсолютные ошибки смотрите в CSV.",
        "",
        f"## Топ-{len(top)} по относительной ошибке",
        report_utils.markdown_table(top_display(top, context_columns)),
        "",
        "## Сводка по выборке",
        report_utils.markdown_table(summary_display(summary)),
        "",
        "## Квантили модуля относительной ошибки",
        report_utils.markdown_table(quantile_display(frame)),
        "",
        "## Наблюдения",
        *[f"- {line}" for line in observations(frame, top, summary)],
    ]
    if dropped:
        lines += ["", f"Строк без пригодного факта или прогноза отброшено: **{dropped}**."]
    if context_note:
        lines += ["", f"Колонки-пояснения не добавлены: {context_note}."]
    lines += [
        "",
        "## Артефакты",
        f"- Каталог: `{report_utils.display_path(output_dir)}`",
        f"- `top_errors.csv` — топ-{len(top)} с относительной ошибкой, остатком"
        + (" и колонками-пояснениями" if context_columns else ""),
        "- `error_summary.csv` — сводные числа в формате `metric,value`",
        "- `figures/top_errors.png` — график по топу",
        "",
        "![Топ объявлений по относительной ошибке](figures/top_errors.png)",
    ]
    return "\n".join(lines) + "\n"


def run(args: argparse.Namespace) -> None:
    if args.top_n < 1:
        raise SystemExit("--top-n должен быть не меньше 1.")
    experiments_dir = args.experiments_dir.expanduser()
    run_dir = resolve_run_dir(args.run_dir, experiments_dir)
    predictions_path = run_dir / "predictions.csv"
    output_dir = args.output_dir.expanduser() if args.output_dir else run_dir / "errors"
    LOG.info("Запуск: %s; результаты: %s", run_dir, output_dir)

    frame, dropped = load_predictions(predictions_path)
    frame = add_error_columns(frame)
    top = frame.nlargest(min(args.top_n, len(frame)), RANK_METRIC).reset_index(drop=True)

    manifest, context, context_note = load_context(run_dir)
    context_columns: list[str] = []
    if not context.empty:
        top = top.merge(context, on="listing_id", how="left")
        context_columns = [column for column in CONTEXT_COLUMNS if column in top.columns]
        if context_columns and not top[context_columns].notna().to_numpy().any():
            context_note = "ни один `listing_id` топа не найден в матрице признаков"

    output_dir.mkdir(parents=True, exist_ok=True)
    top_table = top.copy()
    top_table.insert(0, "rank", range(1, len(top_table) + 1))
    top_path = output_dir / "top_errors.csv"
    top_table.to_csv(top_path, index=False, encoding="utf-8-sig")

    summary = summarize(frame, top)
    summary_path = output_dir / "error_summary.csv"
    summary.to_csv(summary_path, index=False, encoding="utf-8-sig")

    chart = plot_top_errors(top, output_dir / "figures" / "top_errors.png")

    report_path = output_dir / "report.md"
    report_path.write_text(
        build_report(
            run_dir=run_dir, predictions_path=predictions_path, output_dir=output_dir,
            manifest=manifest, frame=frame, top=top, summary=summary,
            context_columns=context_columns, context_note=context_note, dropped=dropped,
        ),
        encoding="utf-8",
    )

    values = dict(zip(summary["metric"], summary["value"]))
    print(f"Строк в выборке: {len(frame)}; в топе: {len(top)}; отброшено: {dropped}")
    print(f"MAPE = {_number(values['mape_pct'], 1)} %; "
          f"медиана |отн. ошибки| = {_number(values['median_abs_rel_error_pct'], 1)} %")
    print(f"Доля топа в абсолютной ошибке: "
          f"{_number(values['top_n_share_of_abs_error_pct'], 1)} %")
    print(f"Отчёт: {report_path}")
    print(f"Результаты: {top_path.name}, {summary_path.name}, figures/{chart.name}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Топ-N ошибок предсказаний (относительная ошибка) из predictions.csv запуска."
    )
    parser.add_argument(
        "--run-dir", type=str, default=None,
        help="Каталог запуска или его номер (по умолчанию — последний номер в --experiments-dir).",
    )
    parser.add_argument(
        "--experiments-dir", type=Path,
        default=report_utils.default_experiments_dir(__file__),
        help="Корень экспериментов (по умолчанию experiments/).",
    )
    parser.add_argument(
        "--top-n", type=int, default=10,
        help="Сколько объявлений попадёт в топ (по умолчанию 10).",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=None,
        help="Куда писать результаты (по умолчанию <каталог запуска>/errors).",
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
    run(args)


if __name__ == "__main__":
    main()
