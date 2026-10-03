#!/usr/bin/env python3
"""Extract text embeddings of listing descriptions with a local Ollama model.

Source:  data/dataset/objects.jsonl (the merged Domclick export).
Output:  data/features/embeddings.csv — `listing_id` plus `emb_0000..emb_NNNN`
         columns, one row per listing (float32 values).
         data/features/embeddings.npy — the same matrix without the id column,
         row order matches the CSV.
         data/features/embeddings_meta.json — model, dimension, counts, timestamp.
         data/features/embeddings_cache.jsonl — resumable per-listing cache.

Model:   qwen3-embedding:8b via the Ollama HTTP API (`POST /api/embed`).

The matrix holds only the join key and embedding columns. For training it is merged
with the structured and LLM feature matrices on `listing_id`:

    structured = pd.read_csv("data/features/features_rich.csv")
    llm = pd.read_csv("data/features/features_llm.csv")
    emb = pd.read_csv("data/features/embeddings.csv")
    matrix = structured.merge(llm, on="listing_id", how="left")
                        .merge(emb, on="listing_id", how="left")

Design rules
------------
1. Only `description` (whitespace-normalized) is embedded; the vector is the text
   representation, not a model prediction, so there is no target leakage.
2. Vectors are L2-normalized by default (qwen3-embedding is trained for cosine
   similarity); disable with `--no-normalize`.
3. Every batch prints a console status line (HTTP code, attempt, latency, dimension,
   outcome). Missing model (HTTP 404) aborts the run with a `ollama pull` hint.
4. Answers are cached per listing (text hash + model + normalize flag), so a run is
   resumable and unchanged descriptions are never re-embedded.
5. Listings without a description get NaN vectors; they are counted in the report
   and in `embeddings_meta.json`.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import requests


LOG = logging.getLogger("domclick.embeddings")

IDENTIFIER_COLUMN = "listing_id"
TEXT_COLUMN = "description"
DEFAULT_HOST = "http://localhost:11434"
DEFAULT_MODEL = "qwen3-embedding:8b"
DIM_PADDING = 4


def default_root() -> Path:
    return Path(__file__).resolve().parent


def default_input_path() -> Path:
    return default_root() / "data" / "dataset" / "objects.jsonl"


def default_features_path() -> Path:
    return default_root() / "data" / "features" / "embeddings.csv"


def default_matrix_path() -> Path:
    return default_root() / "data" / "features" / "embeddings.npy"


def default_meta_path() -> Path:
    return default_root() / "data" / "features" / "embeddings_meta.json"


def default_cache_path() -> Path:
    return default_root() / "data" / "features" / "embeddings_cache.jsonl"


def read_jsonl(path: Path) -> tuple[list[dict[str, Any]], int]:
    """Read JSONL records and skip malformed/non-object lines with a warning."""
    records: list[dict[str, Any]] = []
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
                records.append(record)
    except OSError as exc:
        raise SystemExit(f"Не удалось прочитать {path}: {exc}") from exc

    if not records:
        raise SystemExit(f"В файле нет корректных JSON-объектов: {path}")
    return records, invalid_lines


def description_text(record: dict[str, Any]) -> str:
    """Whitespace-normalized description; empty string when the field is missing."""
    value = record.get(TEXT_COLUMN)
    if value is None:
        return ""
    return " ".join(str(value).split())


def content_hash(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:16]


def request_status(label: str, info: dict[str, Any], attempt: int,
                   attempts_total: int, outcome: str) -> str:
    """Build one console line describing a single embedding request."""
    parts = [f"[{label}]"]
    parts.append(f"HTTP {info['status_code']}" if info.get("status_code") is not None else "сеть")
    parts.append(f"попытка {attempt}/{attempts_total}")
    if info.get("elapsed_s") is not None:
        parts.append(f"{info['elapsed_s']:.1f} с")
    if info.get("dimension"):
        parts.append(f"dim {info['dimension']}")
    parts.append(outcome)
    if info.get("error"):
        parts.append(str(info["error"])[:160])
    return "  ".join(parts)


class OllamaEmbedClient:
    """Minimal Ollama `/api/embed` client with optional L2 normalization."""

    def __init__(self, host: str, model: str, timeout: float, normalize: bool) -> None:
        self.url = host.rstrip("/") + "/api/embed"
        self.model = model
        self.timeout = timeout
        self.normalize = normalize

    def embed(self, texts: list[str]) -> tuple[list[np.ndarray] | None, dict[str, Any]]:
        """Embed a batch of texts; returns (vectors, info) with request metadata."""
        payload: dict[str, Any] = {"model": self.model, "input": list(texts), "truncate": True}
        started = time.monotonic()
        response = requests.post(self.url, json=payload, timeout=self.timeout)
        info: dict[str, Any] = {
            "status_code": response.status_code,
            "elapsed_s": round(time.monotonic() - started, 2),
        }
        try:
            data = response.json()
        except ValueError:
            data = {}
        if not isinstance(data, dict):
            data = {}
        if response.status_code >= 400:
            info["error"] = str(data.get("error") or response.text[:200]).strip()
            return None, info
        raw = data.get("embeddings")
        if not isinstance(raw, list) or len(raw) != len(texts):
            got = len(raw) if isinstance(raw, list) else 0
            info["error"] = f"ожидалось {len(texts)} векторов, получено {got}"
            return None, info
        vectors: list[np.ndarray] = []
        for vector in raw:
            if not isinstance(vector, list) or not vector:
                info["error"] = "пустой вектор в ответе"
                return None, info
            values = np.asarray(vector, dtype="float64")
            if self.normalize:
                norm = float(np.linalg.norm(values))
                if norm > 0:
                    values = values / norm
            vectors.append(values)
        info["dimension"] = int(vectors[0].size)
        return vectors, info


def load_cache(path: Path) -> dict[str, dict[str, Any]]:
    records: dict[str, dict[str, Any]] = {}
    if not path.exists():
        return records
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            item = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(item, dict) and item.get("listing_id") is not None:
            records[str(item["listing_id"])] = item
    return records


def collect_embeddings(records: list[dict[str, Any]], options: argparse.Namespace
                       ) -> tuple[dict[str, list[float]], dict[str, int]]:
    """Embed descriptions in batches, reusing cached vectors when still valid."""
    cache_path = options.cache.expanduser()
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cached = load_cache(cache_path)
    limit = options.limit or len(records)

    vectors: dict[str, list[float]] = {}
    pending: list[tuple[str, str, str]] = []
    stats = {"listings": len(records), "from_cache": 0, "embedded": 0, "empty_text": 0,
             "failed": 0, "beyond_limit": 0, "dimension": 0}

    for position, record in enumerate(records):
        listing_id = str(record.get(IDENTIFIER_COLUMN, position))
        text = description_text(record)
        if not text:
            stats["empty_text"] += 1
            continue
        digest = content_hash(text)
        entry = cached.get(listing_id)
        fresh = (isinstance(entry, dict) and entry.get("text_hash") == digest
                 and entry.get("model") == options.model
                 and entry.get("normalize") == options.normalize
                 and isinstance(entry.get("vector"), list))
        if fresh:
            vectors[listing_id] = entry["vector"]
            stats["from_cache"] += 1
            stats["dimension"] = max(stats["dimension"], len(entry["vector"]))
            continue
        if position >= limit:
            stats["beyond_limit"] += 1
            continue
        pending.append((listing_id, text, digest))

    batches = [pending[index:index + options.batch_size]
               for index in range(0, len(pending), options.batch_size)]
    client = OllamaEmbedClient(options.host, options.model, options.timeout, options.normalize)
    attempts_total = options.retries + 1

    with cache_path.open("a", encoding="utf-8") as cache_file:
        for batch_index, batch in enumerate(batches, start=1):
            label = f"{batch_index}/{len(batches)}"
            texts = [item[1] for item in batch]
            vectors_batch: list[np.ndarray] | None = None
            info: dict[str, Any] = {}
            for attempt in range(1, attempts_total + 1):
                try:
                    vectors_batch, info = client.embed(texts)
                except requests.RequestException as exc:
                    info = {"error": str(exc)}
                    print(request_status(label, info, attempt, attempts_total, "ошибка сети"),
                          flush=True)
                    vectors_batch = None
                    time.sleep(min(30.0, 2.0 ** attempt))
                    continue
                if info.get("status_code") == 404:
                    print(request_status(label, info, attempt, attempts_total, "нет модели"),
                          flush=True)
                    raise SystemExit(
                        f"Ollama не нашла модель {options.model}: "
                        f"{info.get('error', 'модель не загружена')}\n"
                        f"Загрузите её: ollama pull {options.model}"
                    )
                if vectors_batch is None:
                    print(request_status(label, info, attempt, attempts_total, "ошибка"), flush=True)
                    time.sleep(min(30.0, 2.0 ** attempt))
                    continue
                print(request_status(label, info, attempt, attempts_total, "ok"), flush=True)
                break
            if vectors_batch is None:
                stats["failed"] += len(batch)
                LOG.warning("Не удалось получить эмбеддинги для %s объявлений", len(batch))
                continue
            stats["dimension"] = max(stats["dimension"], int(info.get("dimension") or 0))
            for (listing_id, _text, digest), vector in zip(batch, vectors_batch):
                values = [float(value) for value in vector]
                vectors[listing_id] = values
                stats["embedded"] += 1
                cache_file.write(json.dumps({
                    "listing_id": listing_id, "text_hash": digest, "model": options.model,
                    "normalize": options.normalize, "dimension": len(values),
                    "processed_at": datetime.now(timezone.utc).isoformat(),
                    "vector": values,
                }, ensure_ascii=False) + "\n")
            cache_file.flush()

    return vectors, stats


def write_outputs(records: list[dict[str, Any]], vectors: dict[str, list[float]],
                  dimension: int, options: argparse.Namespace, invalid_lines: int,
                  stats: dict[str, int]) -> pd.DataFrame:
    """Write the CSV matrix, the `.npy` array and the metadata JSON."""
    ids = [str(record.get(IDENTIFIER_COLUMN, position))
           for position, record in enumerate(records)]
    columns = [f"emb_{index:0{DIM_PADDING}d}" for index in range(dimension)]
    matrix = np.full((len(ids), dimension), np.nan, dtype="float32")
    for row, listing_id in enumerate(ids):
        vector = vectors.get(listing_id)
        if vector:
            matrix[row, :] = np.asarray(vector, dtype="float32")

    frame = pd.DataFrame(matrix, columns=columns)
    frame.insert(0, IDENTIFIER_COLUMN, ids)

    features_path = options.features_file.expanduser()
    features_path.parent.mkdir(parents=True, exist_ok=True)
    temp = features_path.with_suffix(features_path.suffix + ".tmp")
    frame.to_csv(temp, index=False, encoding="utf-8-sig")
    temp.replace(features_path)

    matrix_path = options.matrix_file.expanduser()
    matrix_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(matrix_path, matrix)

    meta = {
        "model": options.model,
        "host": options.host,
        "dimension": dimension,
        "normalized": options.normalize,
        "listings": len(ids),
        "embedded": int(sum(1 for listing_id in ids if vectors.get(listing_id))),
        "empty_description": stats["empty_text"],
        "from_cache": stats["from_cache"],
        "embedded_in_run": stats["embedded"],
        "failed_batches_listings": stats["failed"],
        "beyond_limit": stats["beyond_limit"],
        "invalid_json_lines": invalid_lines,
        "features_file": str(features_path),
        "matrix_file": str(matrix_path),
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    meta_path = options.meta_file.expanduser()
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    return frame


def run(options: argparse.Namespace) -> None:
    records, invalid_lines = read_jsonl(options.input.expanduser())
    vectors, stats = collect_embeddings(records, options)
    dimension = stats["dimension"]
    if dimension == 0:
        raise SystemExit(
            "Не получено ни одного вектора: проверьте модель Ollama и наличие описаний."
        )
    frame = write_outputs(records, vectors, dimension, options, invalid_lines, stats)

    filled = int((~frame.iloc[:, 1:].isna().all(axis=1)).sum()) if len(frame) else 0
    print(f"Строк: {len(frame)}; колонок: {frame.shape[1]} "
          f"({IDENTIFIER_COLUMN} + {dimension} признаков)")
    print(f"Векторов: {stats['from_cache'] + stats['embedded']} "
          f"(из кэша {stats['from_cache']}, заново {stats['embedded']}); "
          f"строк с вектором {filled}; без описания {stats['empty_text']}; "
          f"ошибок {stats['failed']}; битых строк JSON {invalid_lines}")
    print(f"Признаки: {options.features_file.expanduser()}")
    print(f"Матрица:  {options.matrix_file.expanduser()}")
    print(f"Метаданные: {options.meta_file.expanduser()}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Извлечение эмбеддингов описаний объявлений через Ollama "
                    f"({DEFAULT_MODEL}) для обучения модели цены."
    )
    parser.add_argument(
        "--input", type=Path, default=default_input_path(),
        help="Путь к objects.jsonl (по умолчанию data/dataset/objects.jsonl).",
    )
    parser.add_argument(
        "--features-file", type=Path, default=default_features_path(),
        help="Куда записать матрицу признаков "
             "(по умолчанию data/features/embeddings.csv).",
    )
    parser.add_argument(
        "--matrix-file", type=Path, default=default_matrix_path(),
        help="Куда записать матрицу без id, float32 (по умолчанию "
             "data/features/embeddings.npy).",
    )
    parser.add_argument(
        "--meta-file", type=Path, default=default_meta_path(),
        help="Файл метаданных (по умолчанию data/features/embeddings_meta.json).",
    )
    parser.add_argument(
        "--cache", type=Path, default=default_cache_path(),
        help="Кэш векторов (по умолчанию data/features/embeddings_cache.jsonl).",
    )
    parser.add_argument(
        "--model", default=DEFAULT_MODEL,
        help=f"Модель эмбеддингов Ollama (по умолчанию {DEFAULT_MODEL}).",
    )
    parser.add_argument(
        "--host", default=DEFAULT_HOST,
        help=f"Адрес HTTP API Ollama (по умолчанию {DEFAULT_HOST}).",
    )
    parser.add_argument(
        "--batch-size", type=int, default=16,
        help="Сколько описаний отправлять в одном запросе (по умолчанию 16).",
    )
    parser.add_argument(
        "--limit", type=int, default=0,
        help="Обработать только первые N объявлений; 0 — все. Из кэша берутся все векторы.",
    )
    parser.add_argument(
        "--retries", type=int, default=2,
        help="Дополнительные попытки при ошибке запроса (по умолчанию 2).",
    )
    parser.add_argument(
        "--timeout", type=float, default=600,
        help="Таймаут одного запроса, секунд (по умолчанию 600).",
    )
    normalize_group = parser.add_mutually_exclusive_group()
    normalize_group.add_argument(
        "--normalize", dest="normalize", action="store_true", default=True,
        help="L2-нормировать векторы для косинусной близости (по умолчанию включено).",
    )
    normalize_group.add_argument(
        "--no-normalize", dest="normalize", action="store_false",
        help="Сохранять векторы как вернула модель.",
    )
    parser.add_argument("--verbose", action="store_true",
                        help="Показывать предупреждения и подробные сообщения.")
    args = parser.parse_args()
    if args.batch_size < 1 or args.limit < 0 or args.retries < 0 or args.timeout <= 0:
        parser.error("Некорректные параметры: --batch-size >= 1, --limit/--retries >= 0, "
                     "--timeout > 0")
    return args


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(levelname)s: %(message)s",
    )
    run(args)


if __name__ == "__main__":
    main()
