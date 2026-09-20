from ml_toolkit.arrays import array_max, array_min
import numpy as np


def normalize(arr):
    # Если на вход пришел пустой список, возвращаем его же
    if not arr or len(arr) == 0:
        return arr

    # 1. Вспомогательная функция для нормализации ОДНОГО плоского столбца
    def normalize_column(col):
        c_min = min(col)
        c_max = max(col)
        razn = c_max - c_min

        # Защита от деления на ноль (если все значения в столбце одинаковые)
        if razn == 0:
            return [0.0 for x in col]

        return [(x - c_min) / razn for x in col]

    # 2. Транспонируем матрицу: превращаем список строк в список столбцов
    # Матрица [[10, 100], [20, 200]] превратится в [(10, 20), (100, 200)]
    columns = list(zip(*arr))

    # 3. Нормализуем каждый столбец независимо друг от друга
    normalized_columns = [normalize_column(col) for col in columns]

    # 4. Транспонируем обратно: склеиваем нормализованные столбцы снова в строки
    # Превращаем список нормализованных колонок обратно в формат [[признак1, признак2], ...]
    # map(list, ...) нужен для того, чтобы кортежи после zip снова стали обычными списками
    return list(map(list, zip(*normalized_columns)))
