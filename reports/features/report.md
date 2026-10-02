# Признаки для модели цены

Источник: `C:\Users\developer\projects\domclick\data\dataset\objects.jsonl`
Признаки: **house_area_m2, land_area_sotka**; целевая переменная: **asking_price_rub**.
Записей в источнике: **1,858**; битых строк JSON: **0**.
Осталось в выборке: **1,853**; удалено строк: **5** (0.3 % от источника).

## Сводка по столбцам
| column | role | count | missing | mean | median | std | min | max | skew |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| house_area_m2 | feature | 1 853 | 0 | 318.62 | 237.00 | 295.94 | 15.00 | 3 340.00 | 3.09 |
| land_area_sotka | feature | 1 853 | 0 | 19.80 | 10.00 | 72.40 | 1.00 | 1 629.00 | 13.13 |
| asking_price_rub | target | 1 853 | 0 | 83 300 640.34 | 35 000 000.00 | 189 726 662.42 | 600 000.00 | 3 176 000 000.00 | 9.82 |

## Корреляция признаков с ценой (Спирмен)
| feature | spearman_with_target |
| --- | --- |
| house_area_m2 | 0.86 |
| land_area_sotka | 0.60 |

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
| land_area_sotka_missing | 5 |
| row_dropped_for_missing_feature | 5 |

## Колонки файла
- `listing_id`
- `house_area_m2`
- `land_area_sotka`
- `asking_price_rub`

## Файлы
- Таблица признаков: `C:\Users\developer\projects\domclick\data\features\features.csv`
- Сводка по столбцам: `C:\Users\developer\projects\domclick\reports\features\feature_summary.csv`
- Корреляции: `C:\Users\developer\projects\domclick\reports\features\feature_correlations.csv`
- Отчёт: `C:\Users\developer\projects\domclick\reports\features\report.md`

## Ограничения
- `asking_price_rub` — цена предложения, а не подтверждённая цена сделки.
- Площадь из карточки может расходиться с документами и с текстом объявления.
- Медиана и среднее цены сильно расходятся, `skew` положительный:
  распределение цены скошено, для линейных моделей может понадобиться логарифм цели.
- Признаков всего два; локация, материалы и транспорт не учтены и уходят в остаток.
