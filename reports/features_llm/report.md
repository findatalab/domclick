# Признаки дома, извлечённые из текста объявления (LLM)

Источник: `C:\Users\andrej\projects\finun\domclick\data\dataset\objects.jsonl`
Матрица: `C:\Users\andrej\projects\finun\domclick\data\features\features_llm.csv`
Строк: **1,144**; колонок всего: **118** (`listing_id` и признаки **117**).
Признаки: **23** числовых и **94** бинарных.
Битых строк JSON: **0**.

Матрица содержит только ключ `listing_id` и признаки `llm_*`. Для обучения она соединяется со структурной матрицей `data/features/features_v0.csv` по `listing_id`:

```python
structured = pd.read_csv("data/features/features_v0.csv")
llm = pd.read_csv("data/features/features_llm.csv")
matrix = structured.merge(llm, on="listing_id", how="left")
```

## Группы признаков
| Группа | Колонок |
| --- | --- |
| llm_utilities | 37 |
| llm_construction | 28 |
| llm_house | 16 |
| llm_land | 14 |
| llm_meta | 10 |
| llm_costs | 7 |
| llm_location | 4 |
| llm_documents | 1 |

## Защита от утечки целевой переменной
`price_label` (цена и цена за м²) извлекается только для аудита и **не превращается в признак**: `asking_price_rub` — целевая переменная, и её наличие в тексте сделало бы признак утечкой. При соединении берите цель только из структурной матрицы.

## Что нужно знать про эти признаки
- Значения — это утверждения объявления, а не проверенные факты: модель не подтверждает юридическую чистоту, качество строительства или исправность.
- `llm_house_area_m2`, `llm_floors`, `llm_land_area_sotka` и `llm_distance_to_mkad_km` дублируют поля карточки: используйте их для сверки, а не как независимые признаки.
- `*_present` отмечает поля, значение которых названо в тексте; `llm_status_*_count` и `llm_*_count` — метапризнаки о полноте извлечения.
- Константные колонки (в том числе поля, оставшиеся `unknown`) удалены автоматически.

## Пропуски
Пропуски **не заполняются**: NaN остаётся в матрице, потому что `HistGradientBoostingRegressor`, XGBoost и LightGBM обрабатывают NaN сами, а заполнение до разбиения на фолды переносит информацию из теста в обучение. Для линейных моделей заполните NaN медианой, посчитанной **только на обучающей части** каждого фолда.

Колонки с покрытием ниже 50 % (17):
| Колонка | Покрытие, % | Заполнено | Группа |
| --- | --- | --- | --- |
| llm_heated_area_m2 | 0,262 | 3 | llm_house |
| llm_heating_cost_rub_per_month | 0,437 | 5 | llm_costs |
| llm_renovation_year | 1,049 | 12 | llm_house |
| llm_settlement_fee_rub_per_month | 2,622 | 30 | llm_costs |
| llm_heating_operational | 5,507 | 63 | llm_utilities |
| llm_electricity_phases | 7,255 | 83 | llm_utilities |
| llm_insulation_thickness_mm | 9,790 | 112 | llm_construction |
| llm_wall_thickness_mm | 12,500 | 143 | llm_construction |
| llm_basement_present | 12,675 | 145 | llm_house |
| llm_construction_year | 17,920 | 205 | llm_house |
| llm_house_registration_claimed | 23,252 | 266 | llm_documents |
| llm_year_round_living_claimed | 36,101 | 413 | llm_house |
| llm_electricity_power_kw | 36,713 | 420 | llm_utilities |
| llm_sewage_connected | 38,199 | 437 | llm_utilities |
| llm_bathrooms | 40,559 | 464 | llm_house |
| llm_bedrooms | 43,357 | 496 | llm_house |
| llm_water_connected | 43,881 | 502 | llm_utilities |

## Признаки из текста объявления (LLM)
Провайдер: DeepSeek API (`https://api.deepseek.com`); модель: `deepseek-chat`; версия промпта `1.0`; кэш ответов: `C:\Users\andrej\projects\finun\domclick\data\features\llm_extractions.jsonl`.
Объявлений: **1,144**; обработано моделью: **1,144** (из кэша 1,109, заново 35; пропущено из-за `--llm-limit`: 0).
Ответов, не прошедших проверку схемы и заменённых пустой схемой: **0**. Каждый ответ проверяется функцией `validate_extraction`: обязательные ключи, допустимые статусы (stated/planned/possible/unknown/conflicting), правило «значение без цитаты = null», типы значений и допустимые перечисления. Некорректный JSON повторно запрашивается (`--llm-retries`) с перечнем найденных ошибок.
`price_label` (цена и цена за м²) извлекается только для аудита и **не превращается в признак**: цена — это целевая переменная, и её наличие в тексте сделало бы признак утечкой.

Покрытие и статусы извлечённых полей (полный файл — `llm_field_summary.csv`):
| Поле | Группа | Покрытие, % | stated | planned | possible | unknown | conflicting |
| --- | --- | --- | --- | --- | --- | --- | --- |
| floors | house | 99,913 | 1 143 | 0 | 0 | 1 | 0 |
| location_text | location | 99,388 | 1 137 | 0 | 0 | 7 | 0 |
| land_area_sotka | land | 96,678 | 1 106 | 0 | 0 | 38 | 0 |
| house_area_m2 | house | 94,930 | 1 086 | 0 | 0 | 58 | 0 |
| highway | location | 94,493 | 1 081 | 0 | 0 | 63 | 0 |
| distance_to_mkad_km | location | 91,783 | 1 050 | 0 | 0 | 94 | 0 |
| permitted_use | land | 65,035 | 744 | 0 | 0 | 400 | 0 |
| wall_material | construction | 62,937 | 720 | 0 | 0 | 424 | 0 |
| water_source | utilities | 59,178 | 675 | 0 | 2 | 467 | 0 |
| electricity_connected | utilities | 50,612 | 579 | 0 | 0 | 565 | 0 |
| sewage_type | utilities | 48,077 | 547 | 1 | 2 | 594 | 0 |
| finishing_stage | house | 47,028 | 538 | 0 | 0 | 606 | 0 |
| gas_status | utilities | 45,542 | 518 | 30 | 25 | 571 | 0 |
| water_connected | utilities | 43,881 | 502 | 0 | 0 | 642 | 0 |
| bedrooms | house | 43,357 | 496 | 0 | 0 | 648 | 0 |
| heating_type | utilities | 43,007 | 490 | 1 | 1 | 652 | 0 |
| bathrooms | house | 40,559 | 464 | 0 | 0 | 680 | 0 |
| finishing_details | house | 40,472 | 462 | 0 | 1 | 681 | 0 |
| sewage_connected | utilities | 38,199 | 437 | 0 | 0 | 707 | 0 |
| access_road | location | 38,024 | 435 | 0 | 0 | 709 | 0 |
| electricity_power_kw | utilities | 36,713 | 419 | 1 | 0 | 724 | 0 |
| year_round_living_claimed | house | 36,101 | 413 | 0 | 0 | 731 | 0 |
| foundation_type | construction | 32,255 | 369 | 0 | 0 | 775 | 0 |
| fencing | land | 31,206 | 355 | 2 | 0 | 787 | 0 |
| parking | land | 31,031 | 355 | 0 | 0 | 789 | 0 |

Показаны первые 25 из 50 строк.

В матрицу попадают только числовые и булевы значения, а также one-hot для полей с небольшим словарём; `*_present` отмечает поля, значение которых названо в тексте. Число полей по каждому статусу и размеры массивов (`llm_*_count`) добавлены как метапризнаки.

## Файлы
- Матрица признаков: `C:\Users\andrej\projects\finun\domclick\data\features\features_llm.csv`
- Словарь признаков: `C:\Users\andrej\projects\finun\domclick\reports\features_llm\feature_dictionary.csv`
- Покрытие и статистика: `C:\Users\andrej\projects\finun\domclick\reports\features_llm\feature_coverage.csv`
- Сводка LLM-полей: `C:\Users\andrej\projects\finun\domclick\reports\features_llm\llm_field_summary.csv`
- Кэш LLM-ответов: `C:\Users\andrej\projects\finun\domclick\data\features\llm_extractions.jsonl`
- Отчёт: `C:\Users\andrej\projects\finun\domclick\reports\features_llm\report.md`
