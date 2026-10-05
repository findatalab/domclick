# Контракт эксперимента

Документ фиксирует, как обучающие скрипты (`train_*.py`) сохраняют результаты и какие
поля обязаны заполнять. Контракт реализован в `report_utils.py`, поэтому новые модели и
новые наборы признаков не заводят собственный формат: они вызывают тот же код.

Версия контракта — `report_utils.CONTRACT_VERSION` (сейчас `1`). Несовместимое изменение
схемы требует поднять версию и обновить этот документ.

## 1. Где лежат результаты

Только обучающие запуски пишут в `experiments/`. Отчёты EDA и построения признаков
(`eda.py`, `build_features*.py`, `build_reach_features.py`, `build_llm_extractions.py`,
`build_embeddings.py`) по-прежнему пишут в `reports/<этап>/`.

```
experiments/
  index.csv        реестр всех запусков (одна строка на запуск, добавляется в конец)
  01/              первый запуск
  02/              второй запуск
  ...
```

Номер каталога выдаёт `report_utils.create_run_dir`: он берёт максимальный существующий
числовой каталог и прибавляет единицу, минимум две цифры (`01`, `02`, …, `100`).
Номер выдаётся до обучения, поэтому каталог появляется даже у упавшего запуска — это
намеренно: он остаётся в реестре как след попытки.

## 2. Состав каталога запуска

| Файл | Обязателен | Назначение |
| --- | --- | --- |
| `meta.json` | да | карточка запуска (см. §3) |
| `report.md` | да | человекочитаемый отчёт: тело собирается шаблонами `report_utils` (§6 п. 5), обёртка — `report_utils.render_report` |
| `metrics.csv` | да | канонические метрики в длинном формате (см. §4) |
| `fold_metrics.csv` | да | пофолдовая таблица в формате скрипта: служебные столбцы (`train_size`, `test_size`, `n_iter`, `clipped`, коэффициенты) |
| `metrics_summary.csv` | да | `mean/std/min/max` по каждой метрике (для ансамблей — по каждой модели) |
| `predictions.csv` | да | out-of-fold прогнозы (факт, прогноз, остаток, при необходимости прогнозы членов ансамбля) |
| `residual_diagnostics.csv` | да | ошибка и смещение по квартилям прогноза |
| `figures/actual_vs_predicted.png` | да | факт против прогноза |
| `figures/residuals.png` | да | остатки |
| `feature_importance.csv` | нет | permutation importance (если скрипт её считает) |
| `tuning.csv` | нет | результаты подбора гиперпараметров |
| `feature_importance_llm.csv` | нет | важности LLM-признаков в `train_histgb_reach+llm.py` |

Шапка отчёта, раздел «Ограничения» и список артефактов формируются автоматически из
`meta.json`; скрипт добавляет только содержательные разделы и текст ограничений.

## 3. `meta.json`

```json
{
  "contract_version": 1,
  "run_id": "07",
  "created_at": "2026-10-05T15:39:26+03:00",
  "script": "train_histgb.py",
  "model": "histgb",
  "git": { "commit": "a1b2c3d", "dirty": false },
  "data": {
    "input": "data/features/features_rich.csv",
    "input_sha256": "9f2c…",
    "extra_inputs": { "embeddings": "data/features/embeddings.csv" },
    "rows": 1858,
    "features": 103,
    "target": "asking_price_rub",
    "group_column": "group_locality"
  },
  "cv": { "scheme": "group", "folds": 5, "seed": 42 },
  "params": { "…": "все ключи командной строки запуска, как есть" },
  "environment": { "python": "3.14.5", "pandas": "3.0.6", "numpy": "2.5.3",
                   "scikit_learn": "1.9.1", "matplotlib": "3.11.2" },
  "primary_metric": "rmse_rub",
  "primary_value": 116800000.0,
  "artifacts": ["report.md", "metrics.csv", "figures/actual_vs_predicted.png", "…"],
  "notes": ""
}
```

Обязательные поля и правила:

- `contract_version` — целое, версия схемы.
- `run_id` — строка, совпадает с именем каталога (`"07"`).
- `created_at` — ISO-8601 с часовым поясом.
- `script` — имя файла скрипта (`Path(__file__).name`).
- `model` — короткий идентификатор модели: `lin-reg-2f`, `histgb`, `knn`,
  `histgb-reach-llm`, `histgb-reach-embeds`.
- `data.input`, `data.rows`, `data.features`, `data.target` — обязательны;
  `input_sha256` считается по файлу матрицы, `extra_inputs` — дополнительные входы
  (LLM-признаки, эмбеддинги) в виде `имя → путь`. Пути в `data.input` и
  `data.extra_inputs` хранятся **относительно корня репозитория**, поэтому карточка
  переносима между машинами; в `params` при этом лежит дословный вызов CLI.
- `cv.scheme` — `kfold` или `group`; `folds` и `seed` — целые.
- `params` — **все** ключи CLI запуска (`report_utils.cli_params(args)`), чтобы
  сравнивать запуски одной модели по одинаковым полям.
- `environment` — версии интерпретатора и библиотек.
- `primary_metric`/`primary_value` — метрика-ориентир (`rmse_rub`, среднее по фолдам) и
  её значение; используется для ранжирования.

Карточку собирает `build_manifest`, записывает `write_manifest` (или `finish_run`).
Поля не добавляются вручную «на месте»: расширение схемы идёт через `report_utils`.

## 4. `metrics.csv` и `metrics_summary.csv`

`metrics.csv` — единый длинный формат, ровно эти столбцы в этом порядке:

| Столбец | Тип | Смысл |
| --- | --- | --- |
| `run_id` | str | номер запуска |
| `fold` | int | номер фолда (1..N) |
| `model` | str | имя модели; для ансамбля — `hgb_structured`, `mlp_embeddings`, `ensemble` и т. п. |
| `metric` | str | имя метрики |
| `value` | float | значение |

Обязательный минимум метрик (`report_utils.REQUIRED_METRICS`), все — на рублёвой шкале:

- `r2_rub` — коэффициент детерминации в рублях;
- `rmse_rub` — RMSE в рублях (метрика-ориентир);
- `mae_rub` — MAE в рублях;
- `median_ae_rub` — медианная абсолютная ошибка, рубли;
- `mape_pct` — MAPE, проценты.

Дополнительные метрики разрешены и просто попадают в тот же длинный формат; например
скрипты на лог-цели добавляют `r2_log`. Метрики уровня фолда — единственное, что пишется
в `metrics.csv`; служебные столбцы (`train_size`, `test_size`, `n_iter`, `clipped`)
остаются в скриптовых CSV-артефактах и в отчёт не входят.

`metrics_summary.csv` содержит `metric`, а для ансамблей ещё и `model`, и столбцы
`mean/std/min/max`.

## 5. `experiments/index.csv`

Реестр со строгим порядком столбцов:

`run_id, created_at, script, model, input, input_sha256, rows, features, target,
cv_scheme, folds, seed, primary_metric, primary_value, report`

`report` — относительный путь к отчёту (`07/report.md`). Строка добавляется последней;
файл удобно читать `pandas.read_csv("experiments/index.csv")` и сортировать по
`primary_value`.

## 6. Как добавить новую модель или новые признаки

1. Скрипт `train_*.py` импортирует `report_utils` и **не** заводит собственных
   `format_value`/`markdown_table`/`save_figures`/`summary_table`.
2. Каталог запуска: `run_id, run_dir = report_utils.create_run_dir(args.experiments_dir)`.
3. Метрики: набор столбцов объявляется константой `METRIC_COLUMNS` и проверяется
   `report_utils.ensure_metrics(METRIC_COLUMNS)` — отсутствие обязательной метрики
   останавливает запуск.
4. Артефакты сохраняются в `run_dir`; графики — через `report_utils.save_figures(...)`.
5. Тело отчёта собирается из общих шаблонов `report_utils`, а не пишется в скрипте:
   `body = report_utils.build_body(...)`, куда передаются данные и короткие
   переопределения формулировок. Доступные блоки:

   | Шаблон | Что принимает |
   | --- | --- |
   | `metrics_section(summary, metric_columns, *, model_column=…, model_labels=…)` | сводка `mean ± std` по метрикам; при `model_column` — по одной строке на модель |
   | `per_fold_section(metrics, columns, *, model_column=…, model_labels=…)` | таблица метрик по фолдам |
   | `baseline_section(baseline_rmse, model_rmse, *, model_label=…, note=…)` | сравнение с базовой моделью и процент улучшения |
   | `residual_section(diagnostics, *, subject=…, conclusion=…)` | квартили прогноза + выводы о росте ошибки и смещении |
   | `importance_section(importance, *, title=…, intro=…)` | top-15 permutation importance |
   | `tuning_section(tuning, columns, *, title=…, sort_by=…)` | лучший кандидат каждого фолда |
   | `text_section(title, lines)` | произвольный текстовый блок (скриптовые пояснения) |

   Блоки `importance_section`/`tuning_section` возвращают пустой список для пустых
   датафреймов, поэтому условия `if not importance.empty:` в скриптах не нужны.
   `render_report(title=…, manifest=…, body=…, limitations=…)` добавляет заголовок,
   секцию ограничений и список артефактов.
6. Запуск закрывается `report_utils.finish_run(experiments_dir, run_dir, manifest)` —
   это пишет `meta.json` и добавляет строку в `index.csv`.
7. Новые входные файлы (признаки, эмбеддинги) указываются в `extra_inputs`, новые
   ключи CLI автоматически попадают в `params`.
8. Несовместимое изменение набора полей — поднять `CONTRACT_VERSION` и обновить §3–§5.
