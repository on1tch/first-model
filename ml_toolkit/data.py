import random

def train_test_split(X, y, test_size=0.2, random_state=None):
    # Устанавливаем зерно случайности, если оно передано
    if random_state is not None:
        random.seed(random_state)

    indices = list(range(len(X)))
    random.shuffle(indices)

    X_shuffled = [X[i] for i in indices]
    y_shuffled = [y[i] for i in indices]

    test_count = int(-(len(X) * test_size) // 1 * -1)
    split_index = len(X) - test_count

    X_train = X_shuffled[:split_index]
    X_test = X_shuffled[split_index:]
    y_train = y_shuffled[:split_index]
    y_test = y_shuffled[split_index:]

    return X_train, X_test, y_train, y_test
