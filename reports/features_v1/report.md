# Компактная матрица: числовой каркас и 10 групп признаков

Источник: `C:\Users\andrej\projects\finun\domclick\data\dataset\objects.jsonl`
Извлечения из текста: `C:\Users\andrej\projects\finun\domclick\data\features\llm_extractions.jsonl`
Матрица: `C:\Users\andrej\projects\finun\domclick\data\features\features_v1.csv`
Строк: **1 144**; колонок всего: **84** (признаков **80**, плюс `listing_id`, цель `asking_price_rub`, вариант цели `log_asking_price_rub` и ключ групп `group_locality`).
Признаки: **14** числовых и **66** бинарных (категориальные уровни развёрнуты в колонки-индикаторы).
Битых строк JSON: **0**; строк без положительной цены отброшено: **0**.
Статусы извлечений, значения которых разрешено использовать: `stated`.

## Группы признаков
Числовой каркас берётся из карточки объявления (извлечения заполняют пропуски), остальные группы собираются из ответов модели. У каждой группы свой словарь уровней, поэтому синонимы не разъезжаются по разным колонкам.
| группа | колонок | семейств | среднее покрытие, % |
| --- | --- | --- | --- |
| числовой каркас | 15 | 7 | 98,4 |
| Земля | 8 | 1 | 67 |
| Отделка | 5 | 1 | 47 |
| Газ | 5 | 1 | 45,3 |
| Стены | 8 | 1 | 62,9 |
| Фундамент | 6 | 1 | 32,3 |
| Отопление | 7 | 2 | 24,2 |
| Вода и канализация | 13 | 4 | 47,2 |
| Планировка | 4 | 4 | 27,2 |
| Дополнительные строения | 7 | 7 | 58,7 |
| Возраст | 2 | 2 | 9,5 |

## Признаки и источники значений
Источник «карточка + LLM» означает, что значение карточки приоритетно и только пустые места заполнены извлечением.
| группа | признак | источник | из карточки | из LLM | пропусков | покрытие, % |
| --- | --- | --- | --- | --- | --- | --- |
| числовой каркас | house_area_m2 | карточка + LLM | 1 144 | 0 | 0 | 100 |
| числовой каркас | distance_to_mkad_km | карточка + LLM | 1 056 | 27 | 61 | 94,7 |
| числовой каркас | highway | карточка + LLM | 1 056 | 25 | 63 | 94,5 |
| числовой каркас | land_area_sotka | карточка + LLM | 1 141 | 0 | 3 | 99,7 |
| числовой каркас | floors | карточка + LLM | 1 138 | 5 | 1 | 99,9 |
| числовой каркас | house_area_per_floor | производный | 0 | 0 | 1 | 99,9 |
| числовой каркас | land_to_house_ratio | производный | 0 | 0 | 3 | 99,7 |
| Земля | land_category | карточка + LLM | 681 | 85 | 378 | 67 |
| Отделка | finishing_stage | только LLM | 0 | 538 | 606 | 47 |
| Газ | gas_status | только LLM | 0 | 518 | 626 | 45,3 |
| Стены | wall_material | только LLM | 0 | 720 | 424 | 62,9 |
| Фундамент | foundation_type | только LLM | 0 | 369 | 775 | 32,3 |
| Отопление | heating_type | только LLM | 0 | 490 | 654 | 42,8 |
| Отопление | heating_operational | только LLM | 0 | 63 | 1 081 | 5,5 |
| Вода и канализация | water_source | только LLM | 0 | 675 | 469 | 59 |
| Вода и канализация | water_connected | только LLM | 0 | 502 | 642 | 43,9 |
| Вода и канализация | sewage_type | только LLM | 0 | 547 | 597 | 47,8 |
| Вода и канализация | sewage_connected | только LLM | 0 | 437 | 707 | 38,2 |
| Планировка | bedrooms | только LLM | 0 | 496 | 648 | 43,4 |
| Планировка | bathrooms | только LLM | 0 | 464 | 680 | 40,6 |
| Планировка | attic_present | только LLM | 0 | 139 | 1 005 | 12,2 |
| Планировка | basement_present | только LLM | 0 | 145 | 999 | 12,7 |
| Дополнительные строения | additional_buildings_count | только LLM | 0 | 1 144 | 0 | 100 |
| Дополнительные строения | bathhouse_present | только LLM | 0 | 1 144 | 0 | 100 |
| Дополнительные строения | bathhouse_area_m2 | только LLM | 0 | 53 | 1 091 | 4,6 |
| Дополнительные строения | garage_present | только LLM | 0 | 1 144 | 0 | 100 |
| Дополнительные строения | garage_area_m2 | только LLM | 0 | 47 | 1 097 | 4,1 |
| Дополнительные строения | guest_house_present | только LLM | 0 | 1 144 | 0 | 100 |
| Дополнительные строения | guest_house_area_m2 | только LLM | 0 | 22 | 1 122 | 1,9 |
| Возраст | construction_year | только LLM | 0 | 205 | 939 | 17,9 |
| Возраст | renovation_year | только LLM | 0 | 12 | 1 132 | 1 |

## Канонические уровни групп
Уровни описаны в скрипте (`CATEGORICAL_GROUPS`) и выводятся всегда, даже если уровень пуст: схема матрицы не зависит от выборки. Текст вне словаря попадает в `other` (для групп со свободным текстом) или в `unknown` вместе с отсутствующими значениями; всего значений вне словаря: **0**.
| группа | признак | уровень | строк |
| --- | --- | --- | --- |
| Земля | land_category | izhs | 704 |
| Земля | land_category | dnp | 3 |
| Земля | land_category | snt | 39 |
| Земля | land_category | lph | 7 |
| Земля | land_category | znp | 11 |
| Земля | land_category | selkhoz | 2 |
| Земля | land_category | other | 0 |
| Земля | land_category | unknown | 378 |
| Отделка | finishing_stage | bez_otdelki | 71 |
| Отделка | finishing_stage | predchistovaya | 163 |
| Отделка | finishing_stage | gotovaya | 229 |
| Отделка | finishing_stage | chastichnaya | 75 |
| Отделка | finishing_stage | unknown | 606 |
| Газ | gas_status | podklyuchen | 288 |
| Газ | gas_status | po_granice | 190 |
| Газ | gas_status | ryadom | 26 |
| Газ | gas_status | otsutstvuet | 14 |
| Газ | gas_status | unknown | 626 |
| Стены | wall_material | gazobeton | 139 |
| Стены | wall_material | penobeton | 46 |
| Стены | wall_material | kirpich | 209 |
| Стены | wall_material | karkas | 75 |
| Стены | wall_material | brus | 92 |
| Стены | wall_material | brevno | 50 |
| Стены | wall_material | other | 109 |
| Стены | wall_material | unknown | 424 |
| Фундамент | foundation_type | svayno_rostverkovyy | 34 |
| Фундамент | foundation_type | lentochny | 121 |
| Фундамент | foundation_type | plita | 141 |
| Фундамент | foundation_type | svai | 38 |
| Фундамент | foundation_type | other | 35 |
| Фундамент | foundation_type | unknown | 775 |
| Отопление | heating_type | gazovoe | 152 |
| Отопление | heating_type | elektricheskoe | 155 |
| Отопление | heating_type | pechnoe | 61 |
| Отопление | heating_type | tverdotoplivnoe | 6 |
| Отопление | heating_type | other | 116 |
| Отопление | heating_type | unknown | 654 |
| Вода и канализация | water_source | skvazhina | 321 |
| Вода и канализация | water_source | kolodec | 172 |
| Вода и канализация | water_source | other | 25 |
| Вода и канализация | water_source | centralnoe | 157 |
| Вода и канализация | water_source | privoza | 0 |
| Вода и канализация | water_source | unknown | 469 |
| Вода и канализация | sewage_type | centralnoe | 129 |
| Вода и канализация | sewage_type | septik | 359 |
| Вода и канализация | sewage_type | vygrebnaya_yama | 5 |
| Вода и канализация | sewage_type | other | 54 |
| Вода и канализация | sewage_type | unknown | 597 |

## Уровни шоссе (one-hot по частоте)
Шоссе — часть числового каркаса: уровень получает отдельную колонку, если встречается не реже 10 раз; редкие уровни объединены в `_other`. Константные колонки шоссе удаляются автоматически.
| column | note |
| --- | --- |
| highway_00 | уровень: боровское шоссе (62 строк) |
| highway_01 | уровень: варшавское шоссе (78 строк) |
| highway_02 | уровень: волоколамское шоссе (15 строк) |
| highway_03 | уровень: калужское шоссе (385 строк) |
| highway_04 | уровень: киевское шоссе (287 строк) |
| highway_05 | уровень: минское шоссе (22 строк) |
| highway_06 | уровень: симферопольское шоссе (176 строк) |
| highway_other | редкие уровни (15 шт., 56 строк) |
| highway_unknown | пустое значение |

## Расхождения карточки и извлечений
Строки, где оба источника дали значение, но значения разошлись (для чисел — относительное расхождение больше 1 %). В матрицу попадает значение карточки.
| признак | расхождений |
| --- | --- |
| house_area_m2 | 6 |
| distance_to_mkad_km | 14 |
| highway | 4 |
| land_area_sotka | 23 |
| floors | 4 |

## Статусы извлечений
Промпт различает существующее, запланированное и возможное, поэтому значения со статусом `planned`/`possible` не используются: газ по границе — не подключённый газ. Набор статусов задаётся ключом `--llm-statuses`.
| feature | принято | отброшено по статусу |
| --- | --- | --- |
| permitted_use | 744 | 0 |
| land_category | 150 | 0 |
| finishing_stage | 538 | 0 |
| gas_status | 518 | 3 |
| wall_material | 720 | 0 |
| foundation_type | 369 | 0 |
| heating_type | 490 | 2 |
| water_source | 675 | 2 |
| sewage_type | 547 | 3 |
| bedrooms | 496 | 0 |
| bathrooms | 464 | 0 |
| attic_present | 139 | 0 |
| basement_present | 145 | 0 |
| heating_operational | 63 | 0 |
| water_connected | 502 | 0 |
| sewage_connected | 437 | 0 |
| construction_year | 205 | 0 |
| renovation_year | 12 | 0 |

Версии промпта в файле извлечений: `1.0` (базовый промпт `prompts/prompt_v1.txt`, ожидаемая версия `1.0`).
Строк датасета без ответа модели: **0**.
Ответы присоединяются к объявлениям по `listing_id`. Если выгрузка менялась после извлечения, значения могли устареть — тогда перезапустите `build_llm_extractions.py`, чтобы обновить кэш.

## Особенности матрицы
- Пропуски **не заполняются**: NaN остаётся, потому что `HistGradientBoostingRegressor`, XGBoost и LightGBM обрабатывают NaN сами, а заполнение до разбиения на фолды переносит информацию из теста в обучение. Для линейных моделей заполняйте NaN медианой, посчитанной **только на обучающей части** каждого фолда.
- Флаги различают «нет» (0) и «нет данных» (NaN): у планировки 0 — явное отрицание в тексте (мансарды нет), у строений 0 — модель не нашла такое строение в описании, а NaN — ответа модели по объявлению нет вовсе.
- Признаки, производные от цены (`price_per_m2_*`), не строятся, а метка платформы «Выгодная цена» не используется: она вычисляется из цены.
- Константные колонки, удалённые автоматически (только уровни шоссе): нет. Уровни групп остаются на месте, чтобы схема не менялась между прогонами.

## Как обучать на этой матрице
1. Целевая переменная: `log_asking_price_rub` для скошенного распределения — метрики затем пересчитывайте обратно через `expm1`.
2. Модели: `HistGradientBoostingRegressor` или LightGBM/XGBoost — они нативно работают с NaN.
3. Разбиение: `GroupKFold` с группами из `group_locality` — объявления одного населённого пункта не должны попадать одновременно в обучение и тест.

## Файлы
- Матрица признаков: `C:\Users\andrej\projects\finun\domclick\data\features\features_v1.csv`
- Словарь признаков: `C:\Users\andrej\projects\finun\domclick\reports\features_v1\feature_dictionary.csv`
- Покрытие и статистика: `C:\Users\andrej\projects\finun\domclick\reports\features_v1\feature_coverage.csv`
- Отчёт: `C:\Users\andrej\projects\finun\domclick\reports\features_v1\report.md`
