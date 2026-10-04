#!/usr/bin/env python3
"""Optional MLflow tracking shared by the training scripts of this project.

MLflow is imported inside the functions, so every script keeps working without it:
`--mlflow` then only prints a warning instead of failing. One run records the same
things everywhere, which is what makes experiments comparable here:

* **params** — CLI arguments of the script (matrix, `--cv` group/kfold, folds, seed,
  hyperparameters) plus extras such as the number of rows and features;
* **metrics** — per-fold values logged with `step=<fold>`, so the UI shows the spread
  between folds, plus mean/std/min/max of every metric across folds;
* **tags** — model family, input matrix name and content hash, git commit and dirty
  flag, python/scikit-learn versions;
* **artifacts** — the whole report directory (`report.md`, CV tables, predictions,
  feature importance, tuning, figures), so the UI and the repository stay in sync;
* **model** — the estimator refit on the full matrix, where the script has a single
  estimator to refit.

Where things live
-----------------
* `mlflow.db` — SQLite tracking store at the repository root (default
  `sqlite:///mlflow.db`). Override with `--mlflow-tracking-uri`,
  `MLFLOW_TRACKING_URI` or `.env`.
* `mlartifacts/` — artifact store of the experiment, ignored by git.
* experiment — `domclick` by default, so all models land in one table; `--mlflow-experiment`,
  `MLFLOW_EXPERIMENT_NAME` or `.env` override it.

Usage
-----
    python train_knn.py --mlflow
    mlflow ui --backend-store-uri sqlite:///mlflow.db

Nothing is written unless `--mlflow` is passed, so the default behaviour of every
script is unchanged.
"""

from __future__ import annotations

import argparse
import hashlib
import logging
import os
import platform
import subprocess
from pathlib import Path
from typing import Any

import pandas as pd


LOG = logging.getLogger("domclick.mlflow")

ROOT = Path(__file__).resolve().parent
DEFAULT_EXPERIMENT = "domclick"
DEFAULT_TRACKING_URI = f"sqlite:///{(ROOT / 'mlflow.db').as_posix()}"
DEFAULT_ARTIFACT_DIR = ROOT / "mlartifacts"
TRACKING_URI_ENV = "MLFLOW_TRACKING_URI"
EXPERIMENT_ENV = "MLFLOW_EXPERIMENT_NAME"
ENV_FILE = ROOT / ".env"
INSTALL_HINT = 'pip install "mlflow>=3.16"'

# Columns of the per-fold metrics table that describe the split, not the quality.
PER_FOLD_COLUMNS = ("fold", "train_size", "test_size")


def add_mlflow_args(parser: argparse.ArgumentParser) -> None:
    """Register the shared `--mlflow*` keys; tracking stays off without `--mlflow`."""
    group = parser.add_argument_group("MLflow (опционально)")
    group.add_argument(
        "--mlflow", action="store_true",
        help="Записать запуск (параметры, метрики, артефакты) в MLflow.",
    )
    group.add_argument(
        "--mlflow-tracking-uri", default=None,
        help=f"Хранилище запусков (по умолчанию {DEFAULT_TRACKING_URI}).",
    )
    group.add_argument(
        "--mlflow-experiment", default=None,
        help=f"Имя эксперимента (по умолчанию {DEFAULT_EXPERIMENT}).",
    )
    group.add_argument(
        "--mlflow-run-name", default=None,
        help="Имя запуска (по умолчанию «модель-матрица», например knn-features_v1).",
    )
    group.add_argument(
        "--mlflow-log-model", action=argparse.BooleanOptionalAction, default=True,
        help="Складывать в запуск модель, обученную на всей матрице (где скрипт это умеет).",
    )


def _env_file_value(key: str) -> str | None:
    """Read one key from the project `.env` without adding another dependency."""
    try:
        content = ENV_FILE.read_text(encoding="utf-8")
    except OSError:
        return None
    for line in content.splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        name, _, value = line.partition("=")
        if name.strip() == key:
            return value.strip().strip('"').strip("'")
    return None


def _resolve(explicit: str | None, env_key: str, default: str) -> str:
    """Priority: CLI key, then environment, then `.env`, then the built-in default."""
    return explicit or os.environ.get(env_key) or _env_file_value(env_key) or default


def file_hash(path: Path) -> str:
    """Short content hash of the input matrix, so a run points at its data version."""
    digest = hashlib.sha256()
    try:
        with Path(path).open("rb") as stream:
            for chunk in iter(lambda: stream.read(1 << 20), b""):
                digest.update(chunk)
    except OSError:
        return ""
    return digest.hexdigest()[:12]


def _git_tags() -> dict[str, str]:
    """Commit hash and dirty flag: a run must be reproducible from the code."""
    def git(*arguments: str) -> str:
        try:
            result = subprocess.run(
                ["git", *arguments], capture_output=True, text=True, timeout=10, check=False,
            )
        except (OSError, subprocess.SubprocessError):
            return ""
        return result.stdout.strip()

    return {
        "git_commit": git("rev-parse", "--short", "HEAD") or "unknown",
        "git_dirty": str(bool(git("status", "--porcelain"))).lower(),
    }


def _library_tags() -> dict[str, str]:
    """Versions of the interpreter and the model library, for comparable environments."""
    tags = {"python": platform.python_version()}
    try:
        import sklearn

        tags["sklearn"] = sklearn.__version__
    except ImportError:
        pass
    return tags


def _args_to_params(args: argparse.Namespace) -> dict[str, Any]:
    """CLI arguments as flat params; the MLflow keys themselves are skipped."""
    params: dict[str, Any] = {}
    for name, value in vars(args).items():
        if name.startswith("mlflow") or value is None:
            continue
        if isinstance(value, Path):
            params[name] = str(value)
        elif isinstance(value, (str, int, float, bool)):
            params[name] = value
        else:
            params[name] = str(value)
    return params


def _ensure_experiment(mlflow: Any, name: str, tracking_uri: str) -> None:
    """Create the experiment on first use, keeping its artifacts inside the project."""
    client = mlflow.MlflowClient()
    if client.get_experiment_by_name(name) is None:
        local_store = tracking_uri.startswith(("sqlite:", "file:")) or "://" not in tracking_uri
        client.create_experiment(
            name, artifact_location=DEFAULT_ARTIFACT_DIR.as_uri() if local_store else None,
        )
    mlflow.set_experiment(name)


def _metric_prefix(row: Any) -> str:
    """Scripts that compare several models tag their metrics with a `model` column."""
    value = getattr(row, "model", None)
    return f"{value}_" if isinstance(value, str) and value else ""


def _log_fold_metrics(mlflow: Any, metrics: pd.DataFrame | None) -> None:
    """Per-fold values with `step=fold`, so the UI keeps the spread between folds."""
    if metrics is None or metrics.empty or "fold" not in metrics.columns:
        return
    numeric = [name for name in metrics.columns
               if name not in PER_FOLD_COLUMNS and name != "model"
               and pd.api.types.is_numeric_dtype(metrics[name])]
    for fold, rows in metrics.sort_values("fold").groupby("fold", sort=True):
        values: dict[str, float] = {}
        for row in rows.itertuples(index=False):
            prefix = _metric_prefix(row)
            for name in numeric:
                value = getattr(row, name)
                if pd.notna(value):
                    values[f"{prefix}{name}"] = float(value)
        if values:
            mlflow.log_metrics(values, step=int(fold))


def _log_summary(mlflow: Any, summary: pd.DataFrame | None) -> None:
    """Mean, standard deviation and extremes of every metric across the folds."""
    if summary is None or summary.empty or "metric" not in summary.columns:
        return
    values: dict[str, float] = {}
    for row in summary.itertuples(index=False):
        prefix = _metric_prefix(row)
        for stat in ("mean", "std", "min", "max"):
            value = getattr(row, stat, None)
            if value is not None and pd.notna(value):
                values[f"{prefix}{row.metric}_{stat}"] = float(value)
    if values:
        mlflow.log_metrics(values)


# Types skops may be told to trust: only numeric-library and scikit-learn internals.
TRUSTED_TYPE_PREFIXES = ("numpy.", "sklearn.", "scipy.")

# Formats a serialized model can be saved in, most-preferred first. `skops` is the safe
# default of MLflow 3; `pickle` is the fallback when skops refuses the estimator.
MODEL_VARIANTS = (
    {"name": "model"},
    {"name": "model", "serialization_format": "pickle"},
    {"artifact_path": "model"},
)


def _skops_trusted_types(model: Any) -> list[str] | None:
    """Types skops must be allowed to accept before it can reload our own estimator.

    MLflow 3 serializes scikit-learn models with skops and verifies the dump by loading
    it back. kNN keeps sklearn's KD-tree and a distance-metric class inside the fitted
    estimator, so both are reported as untrusted and the model cannot be saved without
    naming them. Only numeric-library and scikit-learn internals are allowed: anything
    else means the estimator holds something unexpected, and the caller then falls back
    to the pickle format instead of trusting it blindly.
    """
    try:
        import tempfile

        import skops.io

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "model.skops"
            skops.io.dump(model, path)
            untrusted = list(skops.io.get_untrusted_types(file=path))
    except Exception:  # the probe is auxiliary: without it MLflow falls back to pickle
        return None
    if not untrusted or any(not name.startswith(TRUSTED_TYPE_PREFIXES) for name in untrusted):
        return None
    return untrusted


def _log_model(mlflow: Any, model: Any) -> None:
    """Store the estimator so a run can be reloaded without retraining."""
    trusted = _skops_trusted_types(model)
    variants: list[dict[str, Any]] = list(MODEL_VARIANTS)
    if trusted:
        variants.insert(0, {"name": "model", "skops_trusted_types": trusted})
    error: Exception | str | None = None
    for arguments in variants:
        try:
            mlflow.sklearn.log_model(model, **arguments)
            return
        except TypeError as exc:  # another MLflow major version has other keys
            error = exc
        except Exception as exc:  # untrusted types for this variant: try the next format
            error = exc
    LOG.warning("Модель не записана в MLflow: %s", error)


def log_training_run(args: argparse.Namespace, *, model_name: str, input_path: Path,
                     output_dir: Path, extra_params: dict[str, Any] | None = None,
                     extra_metrics: dict[str, float] | None = None,
                     metrics: pd.DataFrame | None = None,
                     summary: pd.DataFrame | None = None, model: Any = None) -> str | None:
    """Record one training run: params, per-fold metrics, summary, artifacts and model.

    Does nothing unless `args.mlflow` is set, so scripts keep their default behaviour.
    Returns the run id, or None when tracking is disabled or unavailable.
    """
    if not getattr(args, "mlflow", False):
        return None
    # MLflow 3 prints an unrelated assistant hint on import; keep the CLI output clean.
    os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")
    try:
        import mlflow
    except ImportError:
        LOG.warning("MLflow не установлен — запуск не записан. Установка: %s", INSTALL_HINT)
        return None

    tracking_uri = _resolve(args.mlflow_tracking_uri, TRACKING_URI_ENV, DEFAULT_TRACKING_URI)
    experiment = _resolve(args.mlflow_experiment, EXPERIMENT_ENV, DEFAULT_EXPERIMENT)
    mlflow.set_tracking_uri(tracking_uri)
    _ensure_experiment(mlflow, experiment, tracking_uri)

    run_name = args.mlflow_run_name or f"{model_name}-{Path(input_path).stem}"
    params = {**_args_to_params(args), **(extra_params or {})}
    with mlflow.start_run(run_name=run_name) as run:
        mlflow.set_tags({
            **_git_tags(), **_library_tags(),
            "model": model_name,
            "matrix": Path(input_path).name,
            "matrix_hash": file_hash(input_path),
        })
        mlflow.log_params(params)
        _log_fold_metrics(mlflow, metrics)
        _log_summary(mlflow, summary)
        if extra_metrics:
            mlflow.log_metrics({name: float(value) for name, value in extra_metrics.items()})
        if output_dir is not None and Path(output_dir).is_dir():
            mlflow.log_artifacts(str(output_dir), artifact_path="report")
        if model is not None and getattr(args, "mlflow_log_model", True):
            _log_model(mlflow, model)
        run_id = run.info.run_id

    print(f"MLflow: run {run_id} → эксперимент «{experiment}» "
          f"({tracking_uri}); UI: mlflow ui --backend-store-uri {tracking_uri}")
    return run_id
