#!/usr/bin/env python3
"""Shared experiment-run and report utilities for every training script.

Contract
--------
Every training run gets its own numbered directory under `experiments/` (`01`, `02`,
...) and writes exactly the same set of files, so runs of different models stay
comparable:

    experiments/<NN>/
        meta.json                 machine-readable run card (the contract)
        report.md                 human-readable report
        metrics.csv               canonical long metrics: run_id, fold, model, metric, value
        metrics_summary.csv       mean/std/min/max per metric (per model where applicable)
        predictions.csv           out-of-fold predictions
        residual_diagnostics.csv  error and bias by prediction quartile
        figures/*.png             actual-vs-predicted and residuals

A registry of all runs is appended to `experiments/index.csv`. The run card and the
registry have a fixed schema; see `docs/experiment_contract.md`. New models or new
feature sets must go through `build_manifest` + `finish_run`, so every run records the
same parameters, data description, environment and artifacts.

The canonical metric names are `r2_rub`, `rmse_rub`, `mae_rub`, `median_ae_rub` and
`mape_pct` (`REQUIRED_METRICS`); scripts may add extras (for example `r2_log`), and
`ensure_metrics` refuses a run that misses a required one.

The report body is assembled from the templates living here — `build_body`,
`metrics_section`, `per_fold_section`, `baseline_section`, `residual_section`,
`importance_section`, `tuning_section`, `text_section` — so a training script passes
its data and small wording overrides, not Markdown layout.
"""

from __future__ import annotations

import hashlib
import json
import logging
import platform
import subprocess
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Sequence

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


LOG = logging.getLogger("domclick.report")

#: Repository root: input paths in the run card are stored relative to it.
REPO_ROOT = Path(__file__).resolve().parent

#: Version of the run-card / registry schema. Bump when fields change incompatibly.
CONTRACT_VERSION = 1

#: Metric used to rank runs and to fill `primary_value` in the registry.
PRIMARY_METRIC = "rmse_rub"

#: Every run must report these metrics on the ruble scale (extras are allowed).
REQUIRED_METRICS = ("r2_rub", "rmse_rub", "mae_rub", "median_ae_rub", "mape_pct")

#: Canonical long-format metrics table written to `metrics.csv`.
UNIFIED_METRIC_COLUMNS = ("run_id", "fold", "model", "metric", "value")

#: Fixed column order of `experiments/index.csv`.
INDEX_COLUMNS = (
    "run_id", "created_at", "script", "model", "input", "input_sha256",
    "rows", "features", "target", "cv_scheme", "folds", "seed",
    "primary_metric", "primary_value", "report",
)


# --------------------------------------------------------------------------- #
# Data collection
# --------------------------------------------------------------------------- #

def default_experiments_dir(script_file: str | Path) -> Path:
    """`<repo>/experiments` next to the calling script."""
    return Path(script_file).resolve().parent / "experiments"


def display_path(path: Path) -> str:
    """Path relative to the repository root, so run cards are machine-independent."""
    try:
        return Path(path).resolve().relative_to(REPO_ROOT).as_posix()
    except (ValueError, OSError):
        return str(path)


def file_sha256(path: Path) -> str:
    """Content hash of an input file, so a run points at its exact data version."""
    digest = hashlib.sha256()
    try:
        with Path(path).open("rb") as stream:
            for chunk in iter(lambda: stream.read(1 << 20), b""):
                digest.update(chunk)
    except OSError:
        return ""
    return digest.hexdigest()


def git_revision() -> tuple[str, bool]:
    """Current commit and whether the tree is dirty; `unknown` outside a repository."""
    def git(*arguments: str) -> str:
        try:
            result = subprocess.run(
                ["git", *arguments], capture_output=True, text=True, timeout=10, check=False,
            )
        except (OSError, subprocess.SubprocessError):
            return ""
        return result.stdout.strip()

    return git("rev-parse", "--short", "HEAD") or "unknown", bool(git("status", "--porcelain"))


def environment_versions() -> dict[str, str]:
    """Interpreter and library versions, for comparable environments."""
    versions = {"python": platform.python_version()}
    for module_name, key in (("pandas", "pandas"), ("numpy", "numpy"),
                             ("sklearn", "scikit_learn"), ("matplotlib", "matplotlib")):
        try:
            module = __import__(module_name)
        except ImportError:  # pragma: no cover - библиотека необязательна
            continue
        versions[key] = str(getattr(module, "__version__", "unknown"))
    return versions


def json_safe(value: Any) -> Any:
    """Convert a value to something `json.dump` accepts (Path, numpy scalars, ...)."""
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def cli_params(args: Any) -> dict[str, Any]:
    """All CLI parameters of a run as a flat JSON-safe dict."""
    return {name: json_safe(value) for name, value in vars(args).items()}


# --------------------------------------------------------------------------- #
# Run allocation, run card and registry
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class RunManifest:
    """Machine-readable run card; the on-disk schema of `meta.json`."""

    contract_version: int
    run_id: str
    created_at: str
    script: str
    model: str
    git_commit: str
    git_dirty: bool
    input: str
    input_sha256: str
    extra_inputs: dict[str, str]
    rows: int
    features: int
    target: str
    group_column: str
    cv_scheme: str
    folds: int
    seed: int
    params: dict[str, Any]
    environment: dict[str, str]
    primary_metric: str
    primary_value: float | None
    artifacts: list[str] = field(default_factory=list)
    notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "contract_version": self.contract_version,
            "run_id": self.run_id,
            "created_at": self.created_at,
            "script": self.script,
            "model": self.model,
            "git": {"commit": self.git_commit, "dirty": self.git_dirty},
            "data": {
                "input": self.input,
                "input_sha256": self.input_sha256,
                "extra_inputs": self.extra_inputs,
                "rows": self.rows,
                "features": self.features,
                "target": self.target,
                "group_column": self.group_column,
            },
            "cv": {"scheme": self.cv_scheme, "folds": self.folds, "seed": self.seed},
            "params": self.params,
            "environment": self.environment,
            "primary_metric": self.primary_metric,
            "primary_value": self.primary_value,
            "artifacts": self.artifacts,
            "notes": self.notes,
        }


def next_run_number(experiments_dir: Path) -> int:
    """One past the highest numeric run directory (01, 02, ...)."""
    numbers = [
        int(child.name) for child in Path(experiments_dir).iterdir()
        if child.is_dir() and child.name.isdigit()
    ] if Path(experiments_dir).is_dir() else []
    return max(numbers, default=0) + 1


def create_run_dir(experiments_dir: Path) -> tuple[str, Path]:
    """Allocate the next numbered run directory; two-digit, e.g. `01`."""
    experiments_dir = Path(experiments_dir)
    experiments_dir.mkdir(parents=True, exist_ok=True)
    while True:
        run_id = f"{next_run_number(experiments_dir):02d}"
        run_dir = experiments_dir / run_id
        try:
            run_dir.mkdir()
        except FileExistsError:  # another run took the number between scan and mkdir
            continue
        return run_id, run_dir


def build_manifest(*, run_id: str, script: str, model: str, input_path: Path,
                   rows: int, features: int, target: str, group_column: str = "",
                   cv_scheme: str = "", folds: int = 0, seed: int = 0,
                   params: dict[str, Any] | None = None,
                   extra_inputs: dict[str, Any] | None = None,
                   artifacts: Iterable[str] = (), primary_metric: str = PRIMARY_METRIC,
                   primary_value: float | None = None, notes: str = "") -> RunManifest:
    """Assemble the run card: data, CV, CLI params, environment, headline metric."""
    commit, dirty = git_revision()
    return RunManifest(
        contract_version=CONTRACT_VERSION,
        run_id=run_id,
        created_at=datetime.now().astimezone().isoformat(timespec="seconds"),
        script=script,
        model=model,
        git_commit=commit,
        git_dirty=dirty,
        input=display_path(input_path),
        input_sha256=file_sha256(Path(input_path)),
        extra_inputs={key: display_path(value) for key, value in (extra_inputs or {}).items()},
        rows=int(rows),
        features=int(features),
        target=target,
        group_column=group_column,
        cv_scheme=cv_scheme,
        folds=int(folds),
        seed=int(seed),
        params=json_safe(params or {}),
        environment=environment_versions(),
        primary_metric=primary_metric,
        primary_value=None if primary_value is None else float(primary_value),
        artifacts=list(artifacts),
        notes=notes,
    )


def write_manifest(run_dir: Path, manifest: RunManifest) -> Path:
    """Write `meta.json` — the machine-readable run card."""
    path = Path(run_dir) / "meta.json"
    path.write_text(
        json.dumps(manifest.to_dict(), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return path


def append_run_index(experiments_dir: Path, manifest: RunManifest) -> Path:
    """Append one row to `experiments/index.csv`, creating the header on first use."""
    path = Path(experiments_dir) / "index.csv"
    row = {column: getattr(manifest, column, "") for column in INDEX_COLUMNS}
    row["report"] = f"{manifest.run_id}/report.md"
    frame = pd.DataFrame([row], columns=list(INDEX_COLUMNS))
    frame.to_csv(path, mode="a", header=not path.exists(), index=False, encoding="utf-8-sig")
    return path


def finish_run(experiments_dir: Path, run_dir: Path, manifest: RunManifest) -> Path:
    """Persist the run card and register the run in `index.csv`."""
    write_manifest(run_dir, manifest)
    return append_run_index(experiments_dir, manifest)


# --------------------------------------------------------------------------- #
# Unified metrics
# --------------------------------------------------------------------------- #

def ensure_metrics(metric_columns: Sequence[str], required: Sequence[str] = REQUIRED_METRICS) -> None:
    """Refuse a run that does not report the metrics required by the contract."""
    missing = [metric for metric in required if metric not in metric_columns]
    if missing:
        raise ValueError(
            "Нарушен контракт метрик: отсутствуют " + ", ".join(missing)
            + f". Обязательный минимум: {', '.join(required)}."
        )


def long_metrics(metrics: pd.DataFrame, metric_columns: Sequence[str], *,
                 run_id: str, model_column: str | None = None,
                 model_name: str = "") -> pd.DataFrame:
    """Canonical long table `run_id, fold, model, metric, value` for `metrics.csv`."""
    rows: list[dict[str, Any]] = []
    for record in metrics.to_dict("records"):
        fold = int(record["fold"])
        model = str(record.get(model_column, "")) if model_column else model_name
        for metric in metric_columns:
            value = record.get(metric)
            if value is not None and pd.notna(value):
                rows.append({"run_id": run_id, "fold": fold, "model": model,
                             "metric": metric, "value": float(value)})
    return pd.DataFrame(rows, columns=list(UNIFIED_METRIC_COLUMNS))


def metric_summary(metrics: pd.DataFrame, metric_columns: Sequence[str],
                   model_column: str | None = None) -> pd.DataFrame:
    """Mean/std/min/max per metric, per model when `model_column` is given."""
    rows: list[dict[str, Any]] = []
    groups = (metrics.groupby(model_column, sort=False) if model_column
              else [(None, metrics)])
    for model, subset in groups:
        for metric in metric_columns:
            values = subset[metric]
            entry: dict[str, Any] = {"metric": metric, "mean": values.mean(),
                                     "std": values.std(), "min": values.min(),
                                     "max": values.max()}
            if model_column:
                entry = {"model": model, **entry}
            rows.append(entry)
    return pd.DataFrame(rows)


def residual_diagnostics(predictions: pd.DataFrame, quartiles: int = 4,
                         actual_column: str = "actual_rub",
                         predicted_column: str = "predicted_rub") -> pd.DataFrame:
    """Error and bias by prediction quartile, to expose heteroscedasticity."""
    frame = predictions.copy()
    frame["prediction_group"] = pd.qcut(
        frame[predicted_column], quartiles,
        labels=[f"Q{index}" for index in range(1, quartiles + 1)],
    )
    frame["residual"] = frame[actual_column] - frame[predicted_column]
    frame["abs_error"] = frame["residual"].abs()
    return (
        frame.groupby("prediction_group", observed=True)
        .agg(
            count=(actual_column, "size"),
            mean_predicted_rub=(predicted_column, "mean"),
            mean_actual_rub=(actual_column, "mean"),
            mean_residual_rub=("residual", "mean"),
            mean_abs_error_rub=("abs_error", "mean"),
        )
        .reset_index()
    )


# --------------------------------------------------------------------------- #
# Report primitives
# --------------------------------------------------------------------------- #

def format_value(value: Any, digits: int = 3) -> str:
    """Readable value: integers as is, floats with spaces as separators."""
    if value is None or (not isinstance(value, str) and pd.isna(value)):
        return "—"
    if isinstance(value, str):
        return value
    if isinstance(value, (int, np.integer)):
        return f"{value:,}".replace(",", " ")
    if abs(float(value)) >= 1000:
        return f"{float(value):,.0f}".replace(",", " ").replace("-", "−")
    return f"{float(value):.{digits}f}".replace(".", ",")


def markdown_table(frame: pd.DataFrame) -> str:
    """Small Markdown table without relying on the optional `tabulate` package."""
    if frame.empty:
        return "_Нет данных._"
    headers = [str(name) for name in frame.columns]
    lines = ["| " + " | ".join(headers) + " |",
             "| " + " | ".join("---" for _ in headers) + " |"]
    for row in frame.itertuples(index=False, name=None):
        lines.append("| " + " | ".join(format_value(value) for value in row) + " |")
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Report body template
# --------------------------------------------------------------------------- #

#: Human-readable metric names used in the body bullets and table headers.
DEFAULT_METRIC_LABELS = {
    "r2_log": "R² (лог)",
    "r2_rub": "R²",
    "rmse_rub": "RMSE",
    "mae_rub": "MAE",
    "median_ae_rub": "Медианная абсолютная ошибка",
    "mape_pct": "MAPE",
}

#: Units appended to a metric bullet; tabular headers carry the unit in the label.
DEFAULT_METRIC_UNITS = {
    "rmse_rub": " ₽",
    "mae_rub": " ₽",
    "median_ae_rub": " ₽",
    "mape_pct": " %",
}

#: Column names of the standard tables (per-fold metrics, diagnostics, tuning).
DEFAULT_TABLE_LABELS = {
    "fold": "Фолд", "model": "Модель", "train_size": "Обучение", "test_size": "Тест",
    "r2_log": "R² (лог)", "r2_rub": "R² (руб.)", "rmse_rub": "RMSE, ₽",
    "mae_rub": "MAE, ₽", "median_ae_rub": "MedAE, ₽", "mape_pct": "MAPE, %",
    "n_iter": "Итераций", "clipped": "Обрезано прогнозов", "n_neighbors": "Соседей",
    "weights": "Вес", "p": "Метрика p", "candidate": "Кандидат",
    "rmse_log_mean": "RMSE (лог)", "learning_rate": "learning_rate",
    "max_leaf_nodes": "max_leaf_nodes", "min_samples_leaf": "min_samples_leaf",
    "l2_regularization": "l2_regularization",
    "feature": "Признак", "importance_mean_r2_log": "Падение R²",
    "prediction_group": "Квартиль прогноза", "count": "Строк",
    "mean_predicted_rub": "Средний прогноз, ₽", "mean_actual_rub": "Средний факт, ₽",
    "mean_residual_rub": "Среднее смещение, ₽", "mean_abs_error_rub": "Средняя абс. ошибка, ₽",
}


def build_body(*blocks: Sequence[str]) -> list[str]:
    """Assemble the report body from sections, one blank line between non-empty blocks."""
    body: list[str] = []
    for block in blocks:
        block = [line for line in block]
        if not block:
            continue
        if body:
            body.append("")
        body.extend(block)
    return body


def text_section(title: str, lines: Sequence[str]) -> list[str]:
    """A `## Title` section; empty lines are dropped and an empty block returns `[]`."""
    lines = [line for line in lines if line]
    return [f"## {title}", *lines] if lines else []


def metrics_section(summary: pd.DataFrame, metric_columns: Sequence[str], *, labels: dict[str, str] | None = None,
                    units: dict[str, str] | None = None, model_column: str | None = None,
                    model_labels: dict[str, str] | None = None, title: str | None = None) -> list[str]:
    """Metrics block: bullets per metric, or one line per model for ensembles."""
    labels = {**DEFAULT_METRIC_LABELS, **(labels or {})}
    units = {**DEFAULT_METRIC_UNITS, **(units or {})}
    if model_column and model_column in summary.columns:
        lines = [title or "## Метрики (среднее по фолдам)", ""]
        for model in summary[model_column].unique():
            subset = summary[summary[model_column] == model].set_index("metric")
            name = (model_labels or {}).get(model, model)
            try:
                lines.append(
                    f"- **{name}**: R² в рублях "
                    f"**{format_value(subset.loc['r2_rub', 'mean'])}**, "
                    f"RMSE **{format_value(subset.loc['rmse_rub', 'mean'])} ₽**, "
                    f"MAE **{format_value(subset.loc['mae_rub', 'mean'])} ₽**."
                )
            except KeyError:  # a model without ruble metrics cannot be summarised
                continue
        return lines

    index = summary.set_index("metric")
    lines = [title or "## Метрики (среднее ± стандартное отклонение по фолдам)", ""]
    for metric in metric_columns:
        if metric not in index.index:
            continue
        row = index.loc[metric]
        lines.append(
            f"- {labels.get(metric, metric)}: "
            f"**{format_value(row['mean'])} ± {format_value(row['std'])}**{units.get(metric, '')}"
        )
    return lines


def per_fold_section(metrics: pd.DataFrame, columns: Sequence[str], *, labels: dict[str, str] | None = None,
                     model_column: str | None = None, model_labels: dict[str, str] | None = None,
                     title: str = "## Метрики по фолдам") -> list[str]:
    """Per-fold table, optionally relabelling the values of the model column."""
    labels = {**DEFAULT_TABLE_LABELS, **(labels or {})}
    table = metrics[list(columns)].rename(columns={column: labels.get(column, column) for column in columns})
    if model_column and model_labels:
        source = labels.get(model_column, model_column)
        if source in table.columns:
            table[source] = table[source].map(lambda value: model_labels.get(value, value))
    return [title, markdown_table(table)]


def baseline_section(*, baseline_rmse: float, model_rmse: float, model_label: str = "Модель",
                     note: str = "Базовая модель предсказывает среднее по обучающей части фолда:",
                     title: str = "## Сравнение с базовой моделью") -> list[str]:
    """Comparison with the mean predictor, shared by every report."""
    improvement = (1 - model_rmse / baseline_rmse) * 100
    return [
        title,
        note,
        f"RMSE **{format_value(baseline_rmse)}** ₽.",
        f"{model_label} снижает RMSE относительно базовой модели на **{improvement:.1f} %**.",
    ]


def error_growth_sentence(diagnostics: pd.DataFrame, *, suffix: str = "") -> str:
    """How much the mean absolute error grows from the cheapest to the dearest quartile."""
    lowest, highest = diagnostics.iloc[0], diagnostics.iloc[-1]
    gap = float(highest["mean_abs_error_rub"]) / float(lowest["mean_abs_error_rub"])
    return (
        f"Средняя абсолютная ошибка растёт от нижнего квартиля прогноза "
        f"(**{format_value(lowest['mean_abs_error_rub'])}** ₽) к верхнему "
        f"(**{format_value(highest['mean_abs_error_rub'])}** ₽) — "
        f"в **{gap:.1f}** раза{suffix}."
    )


def bias_sentence(diagnostics: pd.DataFrame, *, subject: str = "модель") -> str:
    """Describe the bias pattern from the numbers instead of asserting a sign flip."""
    low = float(diagnostics.iloc[0]["mean_residual_rub"])
    high = float(diagnostics.iloc[-1]["mean_residual_rub"])
    subject_start = subject[:1].upper() + subject[1:]
    if low * high < 0:
        return (
            f"Смещение (`факт − прогноз`) меняет знак: **{format_value(low)}** ₽ в нижнем "
            f"квартиле против **{format_value(high)}** ₽ в верхнем, то есть {subject} "
            "переоценивает один конец рынка и недооценивает другой."
        )
    if low >= 0 and high >= 0:
        return (
            f"Смещение (`факт − прогноз`) остаётся положительным во всех квартилях: "
            f"**{format_value(low)}** ₽ в нижнем против **{format_value(high)}** ₽ "
            f"в верхнем. {subject_start} систематически занижает цену, и сильнее всего — "
            "в дорогом сегменте."
        )
    return (
        f"Смещение (`факт − прогноз`) остаётся отрицательным во всех квартилях: "
        f"**{format_value(low)}** ₽ в нижнем против **{format_value(high)}** ₽ "
        f"в верхнем. {subject_start} систематически завышает цену."
    )


def residual_section(diagnostics: pd.DataFrame, *, title: str = "## Диагностика остатков",
                     intro: str = "Средняя абсолютная ошибка и смещение по квартилям прогноза:",
                     subject: str = "модель", growth_suffix: str = "",
                     bias_text: str | None = None, conclusion: str | None = None,
                     growth: bool = True) -> list[str]:
    """Residual table plus the shared error-growth and bias sentences."""
    labels = DEFAULT_TABLE_LABELS
    lines = [title]
    if intro:
        lines.append(intro)
    lines.append(markdown_table(
        diagnostics.rename(columns={column: labels.get(column, column) for column in diagnostics.columns})
    ))
    if growth:
        lines.append(error_growth_sentence(diagnostics, suffix=growth_suffix))
    lines.append(bias_text or bias_sentence(diagnostics, subject=subject))
    if conclusion:
        lines.append(conclusion)
    return lines


def importance_section(importance: pd.DataFrame, *, title: str = "## Важность признаков",
                       intro: str | None = None, top: int = 15) -> list[str]:
    """Permutation-importance table (top `top` rows)."""
    if importance.empty:
        return []
    intro = intro or ("Permutation importance, посчитанная на тестовых фолдах "
                      "(то есть без утечки из обучения). Падение R² на лог-шкале:")
    return [title, intro, markdown_table(importance.head(top))]


def tuning_section(tuning: pd.DataFrame, columns: Sequence[str], *,
                   title: str = "## Подбор гиперпараметров", intro: str | None = None,
                   sort_by: str = "rmse_log_mean", per_fold: bool = True,
                   labels: dict[str, str] | None = None) -> list[str]:
    """Tuning table: the best candidate of every fold, or every candidate as is."""
    if tuning.empty:
        return []
    intro = intro or ("Поиск выполняется на внутренней групповой разбивке обучающей части фолда, "
                      "тестовый фолд в подборе не участвует. Лучший кандидат каждого фолда:")
    rows = (tuning.sort_values(sort_by).groupby("fold").head(1) if per_fold and "fold" in tuning.columns
            else tuning)
    mapping = {**DEFAULT_TABLE_LABELS, **(labels or {})}
    table = rows[list(columns)].rename(columns={column: mapping.get(column, column) for column in columns})
    return [title, intro, markdown_table(table)]


def save_figures(predictions: pd.DataFrame, figure_dir: Path, *,
                 title_prefix: str, actual_column: str = "actual_rub",
                 predicted_column: str = "predicted_rub") -> list[Path]:
    """Out-of-fold diagnostics: predicted vs actual and residuals (shared layout)."""
    figure_dir = Path(figure_dir)
    figure_dir.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "DejaVu Sans"],
        "axes.unicode_minus": False,
        "figure.dpi": 120,
    })
    actual = predictions[actual_column].to_numpy(dtype=float)
    predicted = predictions[predicted_column].to_numpy(dtype=float)
    outputs: list[Path] = []

    # Values and axis limits must share one unit; mixing rubles with millions clips
    # every point and leaves an empty-looking chart.
    limit = float(np.quantile(np.concatenate([actual, predicted]), 0.99)) / 1e6
    fig, ax = plt.subplots(figsize=(9, 7))
    ax.scatter(actual / 1e6, predicted / 1e6, alpha=0.4, s=18,
               color="#2563eb", edgecolors="none")
    ax.plot([0, limit], [0, limit], color="#dc2626", linewidth=1.4,
            label="идеальное совпадение")
    ax.set(title=f"Факт и прогноз цены, {title_prefix} (out-of-fold)",
           xlabel="Факт, млн руб.", ylabel="Прогноз, млн руб.",
           xlim=(0, limit), ylim=(0, limit))
    ax.legend()
    ax.grid(alpha=0.25)
    fig.tight_layout()
    path = figure_dir / "actual_vs_predicted.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    outputs.append(path)

    residuals = (actual - predicted) / 1e6
    cap = float(np.quantile(np.abs(residuals), 0.99))
    predicted_mln = predicted / 1e6
    x_cap = float(np.quantile(predicted_mln, 0.99))
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.scatter(predicted_mln, residuals, alpha=0.4, s=18,
               color="#7c3aed", edgecolors="none")
    ax.axhline(0, color="#dc2626", linewidth=1.4)
    ax.set(title=f"Остатки: {title_prefix} (out-of-fold)",
           xlabel="Прогноз, млн руб.", ylabel="Остаток, млн руб.",
           xlim=(min(predicted_mln.min(), 0), x_cap), ylim=(-cap, cap))
    ax.grid(alpha=0.25)
    fig.tight_layout()
    path = figure_dir / "residuals.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    outputs.append(path)
    return outputs


def render_report(*, title: str, manifest: RunManifest, body: Sequence[str],
                  limitations: Sequence[str] = ()) -> str:
    """Assemble a report with the shared header, body, limitations and artifact list."""
    lines: list[str] = [f"# {title}", ""]
    lines += _manifest_lines(manifest)
    if body:
        lines += ["", *body]
    if limitations:
        lines += ["", "## Ограничения", *limitations]
    lines += ["", "## Артефакты запуска", f"- Каталог запуска: `{manifest.run_id}/`"]
    lines += [f"- `{artifact}`" for artifact in manifest.artifacts]
    lines.append(f"- Карточка запуска: `{manifest.run_id}/meta.json`; реестр: `index.csv`")
    return "\n".join(lines) + "\n"


def _manifest_lines(manifest: RunManifest) -> list[str]:
    """Standard header of every report; mirrors the run card."""
    lines = [
        f"- Запуск: **{manifest.run_id}** (каталог `{manifest.run_id}/`), "
        f"создан {manifest.created_at}",
        f"- Скрипт: `{manifest.script}`; модель: **{manifest.model}**",
        f"- Данные: `{manifest.input}` (sha256 `{manifest.input_sha256[:12]}`), "
        f"строк: **{manifest.rows:,}**, признаков: **{manifest.features}**",
        f"- Цель: `{manifest.target}`"
        + (f"; ключ групп: `{manifest.group_column}`" if manifest.group_column else ""),
        f"- Валидация: `{manifest.cv_scheme}` ({manifest.folds} фолдов, seed {manifest.seed})",
        f"- Метрика-ориентир: **{manifest.primary_metric}** = "
        f"{format_value(manifest.primary_value)}",
    ]
    if manifest.extra_inputs:
        joined = ", ".join(f"{key}: `{value}`" for key, value in manifest.extra_inputs.items())
        lines.append(f"- Дополнительные входы: {joined}")
    environment = ", ".join(f"{key} {value}" for key, value in manifest.environment.items())
    lines.append(f"- Окружение: {environment}")
    if manifest.git_commit:
        dirty = " (есть незакоммиченные изменения)" if manifest.git_dirty else ""
        lines.append(f"- Код: commit `{manifest.git_commit}`{dirty}")
    return lines
