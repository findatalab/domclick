# Компактная матрица v2: числовой каркас и группы признаков

Источник: `C:\Users\developer\projects\domclick\data\dataset\objects.jsonl`
Матрица: `C:\Users\developer\projects\domclick\data\features\features_v2.csv`
Кэш ответов модели: `C:\Users\developer\projects\domclick\data\features\llm_v2_extractions.jsonl`
Промпты: `prompt_v2_cat.txt` (sha `898ee34d`), `prompt_v2_num.txt` (sha `6b49b81e`)
Модель: `qwen3.8` на `http://127.0.0.1:11434`; версия промпта `2.0`.
Строк: **1 144**; колонок всего: **66** (признаков **62**, плюс `listing_id`, цель `asking_price_rub`, вариант цели `log_asking_price_rub` и ключ групп `group_locality`).
Признаки: **12** числовых и **50** бинарных (категориальные уровни развёрнуты в колонки-индикаторы).
Битых строк JSON: **0**; строк без положительной цены отброшено: **0**.

## Извлечения локальной модели
Извлечения запрашиваются у локального сервера Ollama отдельными промптами для категориальных и для числовых признаков; карточка объявления передаётся как контекст, а цена и метки платформы — нет: они связаны с целевой переменной, и признаки, собранные из них, были бы утечкой.
Ответов в кэше: **1 144**; взято из кэша без запроса: **24**; запрошено в этом запуске: **1 120**; объявлений с ошибкой ответа: **0**; время запросов: **313.6 мин**.
Объявлений с хотя бы одним ответом модели: **1 144** из **1 144**; у остальных признаки групп остаются NaN, а не нулём.

### Заполненность полей извлечения
| поле | признак | заполнено | пусто | доля, % |
| --- | --- | --- | --- | --- |
| land_category | land_category | 1 144 | 0 | 100 |
| permitted_use | permitted_use | 0 | 1 144 | 0 |
| finishing_stage | finishing_stage | 1 144 | 0 | 100 |
| gas_status | gas_status | 1 144 | 0 | 100 |
| wall_material | wall_material | 1 144 | 0 | 100 |
| foundation_type | foundation_type | 1 144 | 0 | 100 |
| heating_type | heating_type | 1 144 | 0 | 100 |
| water_source | water_source | 1 144 | 0 | 100 |
| sewage_type | sewage_type | 1 144 | 0 | 100 |
| bedrooms_count | bedrooms | 740 | 404 | 64,7 |
| bathrooms_count | bathrooms | 655 | 489 | 57,3 |
| has_attic | attic_present | 1 144 | 0 | 100 |
| has_basement | basement_present | 1 144 | 0 | 100 |
| has_bathhouse | bathhouse_present | 1 144 | 0 | 100 |
| has_garage | garage_present | 1 144 | 0 | 100 |
| has_guesthouse | guest_house_present | 1 144 | 0 | 100 |
| bathhouse_area | bathhouse_area_m2 | 61 | 1 083 | 5,3 |
| garage_area | garage_area_m2 | 64 | 1 080 | 5,6 |
| guesthouse_area | guest_house_area_m2 | 44 | 1 100 | 3,8 |
| construction_year | house_age | 309 | 835 | 27 |
| house_age | house_age | 311 | 833 | 27,2 |

## Группы признаков
Числовой каркас берётся из карточки объявления (извлечения заполняют пропуски), остальные группы собираются из ответов модели. У каждой группы свой словарь уровней, поэтому синонимы не разъезжаются по разным колонкам.
| группа | колонок | семейств | среднее покрытие, % |
| --- | --- | --- | --- |
| числовой каркас | 15 | 7 | 98,5 |
| Земля | 5 | 1 | 100 |
| Отделка | 2 | 1 | 100 |
| Газ | 3 | 1 | 100 |
| Стены | 8 | 1 | 100 |
| Фундамент | 6 | 1 | 100 |
| Отопление | 5 | 1 | 100 |
| Вода | 4 | 1 | 100 |
| Канализация | 3 | 1 | 100 |
| Планировка | 4 | 4 | 80,4 |
| Дополнительные строения | 6 | 6 | 52,5 |
| Возраст | 1 | 1 | 27,2 |

## Признаки и источники значений
Источник «карточка + LLM» означает, что значение карточки приоритетно и только пустые места заполнены извлечением.
| группа | признак | источник | из карточки | из LLM | пропусков | покрытие, % |
| --- | --- | --- | --- | --- | --- | --- |
| числовой каркас | house_area_m2 | карточка + LLM | 1 144 | 0 | 0 | 100 |
| числовой каркас | distance_to_mkad_km | карточка + LLM | 1 056 | 26 | 62 | 94,6 |
| числовой каркас | highway | карточка + LLM | 1 056 | 40 | 48 | 95,8 |
| числовой каркас | land_area_sotka | карточка + LLM | 1 141 | 0 | 3 | 99,7 |
| числовой каркас | floors | карточка + LLM | 1 138 | 5 | 1 | 99,9 |
| числовой каркас | house_area_per_floor | производный | 0 | 0 | 1 | 99,9 |
| числовой каркас | land_to_house_ratio | производный | 0 | 0 | 3 | 99,7 |
| Земля | land_category | карточка + LLM | 681 | 463 | 0 | 100 |
| Отделка | finishing_stage | только LLM | 0 | 1 144 | 0 | 100 |
| Газ | gas_status | только LLM | 0 | 1 144 | 0 | 100 |
| Стены | wall_material | только LLM | 0 | 1 144 | 0 | 100 |
| Фундамент | foundation_type | только LLM | 0 | 1 144 | 0 | 100 |
| Отопление | heating_type | только LLM | 0 | 1 144 | 0 | 100 |
| Вода | water_source | только LLM | 0 | 1 144 | 0 | 100 |
| Канализация | sewage_type | только LLM | 0 | 1 144 | 0 | 100 |
| Планировка | bedrooms | только LLM | 0 | 739 | 405 | 64,6 |
| Планировка | bathrooms | только LLM | 0 | 651 | 493 | 56,9 |
| Планировка | attic_present | только LLM | 0 | 1 144 | 0 | 100 |
| Планировка | basement_present | только LLM | 0 | 1 144 | 0 | 100 |
| Дополнительные строения | bathhouse_present | только LLM | 0 | 1 144 | 0 | 100 |
| Дополнительные строения | bathhouse_area_m2 | только LLM | 0 | 61 | 1 083 | 5,3 |
| Дополнительные строения | garage_present | только LLM | 0 | 1 144 | 0 | 100 |
| Дополнительные строения | garage_area_m2 | только LLM | 0 | 64 | 1 080 | 5,6 |
| Дополнительные строения | guest_house_present | только LLM | 0 | 1 144 | 0 | 100 |
| Дополнительные строения | guest_house_area_m2 | только LLM | 0 | 44 | 1 100 | 3,8 |
| Возраст | house_age | только LLM | 0 | 311 | 833 | 27,2 |

## Канонические уровни групп
Уровни описаны в скрипте (`CATEGORICAL_GROUPS`) и выводятся всегда, даже если уровень пуст: схема матрицы не зависит от выборки. Текст вне словаря попадает в уровень `other` (если он есть у группы) либо в тот уровень, куда словарь относит неизвестное значение; всего таких значений: **249**.
| признак | уровни | неизвестное | вне словаря |
| --- | --- | --- | --- |
| land_category | snt, lph, izhs, other | unknown | 0 |
| finishing_stage | without_repair, with_repair | without_repair | 0 |
| gas_status | po_granice, podklyuchen, otsutstvuet | otsutstvuet | 227 |
| wall_material | gazobeton, penobeton, kirpich, karkas, brus, brevno, other | unknown | 0 |
| foundation_type | svayno_rostverkovyy, lentochny, plita, svai, other | unknown | 0 |
| heating_type | gazgolder, gazovoe, elektricheskoe, pechnoe, otsutstvuet | otsutstvuet | 8 |
| water_source | otsutstvuet, skvazhina, kolodec, centralnoe | otsutstvuet | 0 |
| sewage_type | septik, centralnoe, otsutstvuet | otsutstvuet | 14 |

| группа | признак | уровень | строк |
| --- | --- | --- | --- |
| Земля | land_category | snt | 275 |
| Земля | land_category | lph | 7 |
| Земля | land_category | izhs | 720 |
| Земля | land_category | other | 3 |
| Земля | land_category | unknown | 139 |
| Отделка | finishing_stage | without_repair | 524 |
| Отделка | finishing_stage | with_repair | 620 |
| Газ | gas_status | po_granice | 221 |
| Газ | gas_status | podklyuchen | 396 |
| Газ | gas_status | otsutstvuet | 527 |
| Стены | wall_material | gazobeton | 137 |
| Стены | wall_material | penobeton | 47 |
| Стены | wall_material | kirpich | 192 |
| Стены | wall_material | karkas | 71 |
| Стены | wall_material | brus | 110 |
| Стены | wall_material | brevno | 60 |
| Стены | wall_material | other | 113 |
| Стены | wall_material | unknown | 414 |
| Фундамент | foundation_type | svayno_rostverkovyy | 51 |
| Фундамент | foundation_type | lentochny | 118 |
| Фундамент | foundation_type | plita | 133 |
| Фундамент | foundation_type | svai | 37 |
| Фундамент | foundation_type | other | 21 |
| Фундамент | foundation_type | unknown | 784 |
| Отопление | heating_type | gazgolder | 19 |
| Отопление | heating_type | gazovoe | 332 |
| Отопление | heating_type | elektricheskoe | 193 |
| Отопление | heating_type | pechnoe | 134 |
| Отопление | heating_type | otsutstvuet | 466 |
| Вода | water_source | otsutstvuet | 394 |
| Вода | water_source | skvazhina | 311 |
| Вода | water_source | kolodec | 164 |
| Вода | water_source | centralnoe | 275 |
| Канализация | sewage_type | septik | 368 |
| Канализация | sewage_type | centralnoe | 211 |
| Канализация | sewage_type | otsutstvuet | 565 |

## Уровни шоссе (one-hot по частоте)
Шоссе — часть числового каркаса: уровень получает отдельную колонку, если встречается не реже 10 раз; редкие уровни объединены в `highway_other`, отсутствие значения — в `highway_unknown`. Значения карточки и извлечения приводятся к одному виду: слово «шоссе» отбрасывается.
| column | note |
| --- | --- |
| highway_00 | уровень: боровское (63 строк) |
| highway_01 | уровень: варшавское (78 строк) |
| highway_02 | уровень: волоколамское (15 строк) |
| highway_03 | уровень: калужское (389 строк) |
| highway_04 | уровень: киевское (288 строк) |
| highway_05 | уровень: минское (23 строк) |
| highway_06 | уровень: симферопольское (176 строк) |
| highway_other | редкие уровни (24 шт., 64 строк) |
| highway_unknown | шоссе не указано ни в карточке, ни в извлечении |

## Проверка диапазонов
Значения вне осмысленного диапазона заменены на NaN: это ошибки модели, а не измерения. Возраст дома считается как 2026 минус год постройки; год взят из поля `construction_year` (**309** строк), а если модель вернула только возраст — из `house_age` (**311** строк).
| признак | допустимо | отброшено | минимум LLM | максимум LLM |
| --- | --- | --- | --- | --- |
| house_area_m2 | 5 … 20000 | 0 | 15 | 820 |
| distance_to_mkad_km | 0 … 500 | 0 | 1 | 105 |
| land_area_sotka | 0.1 … 5000 | 0 | 1 | 700 |
| floors | 1 … 10 | 0 | 1 | 4 |
| bedrooms | 1 … 20 | 1 | 1 | 10 |
| bathrooms | 1 … 20 | 4 | 1 | 8 |
| attic_present | — | 0 | 0 | 1 |
| basement_present | — | 0 | 0 | 1 |
| bathhouse_area_m2 | 1 … 1000 | 0 | 2,5 | 120 |
| garage_area_m2 | 1 … 1000 | 0 | 15 | 270 |
| guest_house_area_m2 | 1 … 2000 | 0 | 20 | 285 |
| house_age | 0 … 200 | 0 | 0 | 76 |

## Расхождения карточки и извлечений
Строки, где оба источника дали значение, но значения разошлись (для чисел — относительное расхождение больше 1 %). В матрицу попадает значение карточки.
| признак | расхождений |
| --- | --- |
| house_area_m2 | 50 |
| distance_to_mkad_km | 179 |
| highway | 13 |
| land_area_sotka | 68 |
| floors | 18 |

## Особенности матрицы
- Пропуски **не заполняются**: NaN остаётся, потому что `HistGradientBoostingRegressor`, XGBoost и LightGBM обрабатывают NaN сами, а заполнение до разбиения на фолды переносит информацию из теста в обучение. Для линейных моделей заполняйте NaN медианой, посчитанной **только на обучающей части** каждого фолда.
- Группы газ, отопление, вода, канализация и отделка не имеют уровня «неизвестно»: как и требует словарь, неупомянутый или нераспознанный ответ попадает в отрицательный уровень (`otsutstvuet`, `without_repair`). Строки, по которым модель вообще не отвечала, остаются NaN — «не спрашивали» не становится «нет».
- Планировка и дополнительные строения идут и флагом (есть/нет), и числом: 0 — модель явно не нашла признак в тексте, NaN — ответа по объявлению нет.
- Возраст дома — единственная колонка группы «Возраст»: год постройки восстанавливается как `2026 - house_age`, отдельная колонка была бы линейно зависимой копией.
- Признаки, производные от цены (`price_per_m2_*`), не строятся, а метка платформы «Выгодная цена» не используется: она вычисляется из цены.
- Константные колонки, удалённые автоматически (только уровни шоссе): нет. Уровни групп остаются на месте, чтобы схема не менялась между прогонами.

## Как обучать на этой матрице
1. Целевая переменная: `log_asking_price_rub` для скошенного распределения — метрики затем пересчитывайте обратно через `expm1`.
2. Модели: `HistGradientBoostingRegressor` или LightGBM/XGBoost — они нативно работают с NaN.
3. Разбиение: `GroupKFold` с группами из `group_locality` — объявления одного населённого пункта не должны попадать одновременно в обучение и тест.

## Файлы
- Матрица признаков: `C:\Users\developer\projects\domclick\data\features\features_v2.csv`
- Кэш ответов модели: `C:\Users\developer\projects\domclick\data\features\llm_v2_extractions.jsonl`
- Словарь признаков: `C:\Users\developer\projects\domclick\reports\features_v2\feature_dictionary.csv`
- Покрытие и статистика: `C:\Users\developer\projects\domclick\reports\features_v2\feature_coverage.csv`
- Сводка по полям извлечения: `C:\Users\developer\projects\domclick\reports\features_v2\extraction_summary.csv`
- Уровни групп: `C:\Users\developer\projects\domclick\reports\features_v2\feature_levels.csv`
- Проверка диапазонов: `C:\Users\developer\projects\domclick\reports\features_v2\value_ranges.csv`
- Расхождения источников: `C:\Users\developer\projects\domclick\reports\features_v2\disagreements.csv`
- Отчёт: `C:\Users\developer\projects\domclick\reports\features_v2\report.md`
