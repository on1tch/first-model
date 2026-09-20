from ml_toolkit.arrays import array_sum_and_cnt


def mse(y_true, y_pred):
    def get_squares(true_vals, pred_vals):
        squares = []
        for i, j in zip(true_vals, pred_vals):
            if isinstance(i, list) and isinstance(j, list):
                squares.extend(get_squares(i, j))
            elif isinstance(i, (int, float)):
                squares.append((i - j) ** 2)
        return squares

    all_squares = get_squares(y_true, y_pred)

    if not all_squares:
        return 0.0
    mse = sum(all_squares) / len(all_squares)
    return mse


def mae(y_true, y_pred):
    def get_differences(true_vals, pred_vals):
        differences = []
        for i, j in zip(true_vals, pred_vals):
            if isinstance(i, list) and isinstance(j, list):
                differences.extend(get_differences(i, j))
            elif isinstance(i, (int, float)):
                differences.append(abs(i - j))
        return differences

    all_squares = get_differences(y_true, y_pred)

    if not all_squares:
        return 0.0
    summ, cnt = array_sum_and_cnt(all_squares)
    return summ / cnt


def accuracy(y_true, y_pred):
    def get_accuracies(true_vals, pred_vals):
        accuracies = []
        for i, j in zip(true_vals, pred_vals):
            # Если оба элемента списки — спускаемся глубже
            if isinstance(i, list) and isinstance(j, list):
                accuracies.extend(get_accuracies(i, j))
            # Если оба элемента НЕ списки — безопасно сравниваем их напрямую
            elif not isinstance(i, list) and not isinstance(j, list):
                accuracies.append(i == j)
        return accuracies

    accuracies = get_accuracies(y_true, y_pred)

    if not accuracies:
        return 0.0

    return accuracies.count(True) / len(accuracies)
