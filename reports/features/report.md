# Признаки для модели цены

Источник: `C:\Users\andrej\projects\finun\domclick\data\dataset\objects.jsonl`
Признаки: **house_area_m2, land_area_sotka**; целевая переменная: **asking_price_rub**.
Записей в источнике: **1,144**; битых строк JSON: **0**.
Осталось в выборке: **1,141**; удалено строк: **3** (0.3 % от источника).

## Сводка по столбцам
| column | role | count | missing | mean | median | std | min | max | skew |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| house_area_m2 | feature | 1 141 | 0 | 181.13 | 151.00 | 118.79 | 15.00 | 820.00 | 1.56 |
| land_area_sotka | feature | 1 141 | 0 | 15.03 | 8.00 | 64.79 | 1.00 | 1 005.00 | 11.28 |
| asking_price_rub | target | 1 141 | 0 | 21 961 772.41 | 19 800 000.00 | 12 873 132.63 | 600 000.00 | 50 000 000.00 | 0.47 |

## Корреляция признаков с ценой (Спирмен)
| feature | spearman_with_target |
| --- | --- |
| house_area_m2 | 0.77 |
| land_area_sotka | 0.34 |

## Проверки качества данных
Число строк по каждой проверке; одна строка может попасть в несколько проверок.
| Проверка | Строк |
| --- | --- |
| target_missing_or_not_positive | 0 |
| house_area_m2_negative | 0 |
| house_area_m2_zero | 0 |
| land_area_sotka_negative | 0 |
| land_area_sotka_zero | 0 |
| house_area_m2_missing | 0 |
| land_area_sotka_missing | 3 |
| row_dropped_for_missing_feature | 3 |

## Колонки файла
- `listing_id`
- `house_area_m2`
- `land_area_sotka`
- `asking_price_rub`

## Файлы
- Таблица признаков: `C:\Users\andrej\projects\finun\domclick\data\features\features.csv`
- Сводка по столбцам: `C:\Users\andrej\projects\finun\domclick\reports\features\feature_summary.csv`
- Корреляции: `C:\Users\andrej\projects\finun\domclick\reports\features\feature_correlations.csv`
- Отчёт: `C:\Users\andrej\projects\finun\domclick\reports\features\report.md`

## Ограничения
- `asking_price_rub` — цена предложения, а не подтверждённая цена сделки.
- Площадь из карточки может расходиться с документами и с текстом объявления.
- Медиана и среднее цены сильно расходятся, `skew` положительный:
  распределение цены скошено, для линейных моделей может понадобиться логарифм цели.
- Признаков всего два; локация, материалы и транспорт не учтены и уходят в остаток.
