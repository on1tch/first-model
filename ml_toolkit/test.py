from math import exp
import random
import numpy as np
from data import train_test_split
import matplotlib.pyplot as plt


random.seed(42)
num_points = 1000
X = []
y = []

for _ in range(num_points):
    height = random.uniform(150, 200)
    weight = random.uniform(45, 110)
    score = 0.08 * height + 0.04 * weight

    if score > 18:
        label = 1
    else:
        label = 0

    X.append([height,weight])
    y.append(label)

def accuracy(y_true, y_pred):
    return np.mean(y_true == y_pred)
def normalize_column(values):
    min_value = min(values)
    max_value = max(values)
    return [(x - min_value) / (max_value - min_value) for x in values]


heights = normalize_column([row[0] for row in X])
weights = normalize_column([row[1] for row in X])

X = [[h, w] for h, w in zip(heights, weights)]
X_train_list, X_test_list, y_train_list, y_test_list = train_test_split(X, y)

X_train = np.array(X_train_list)
X_test = np.array(X_test_list)
y_train = np.array(y_train_list)
y_test = np.array(y_test_list)

def predict(X, w1, w2, b):
    X1, X2 = X[:, 0], X[:, 1]
    z = w1 * X1 + w2 * X2 + b
    probabilities = 1 / (1 + np.exp(-z))
    predictions = (probabilities >= 0.5).astype(int)
    return predictions
def calculate_log_loss(X, y_true, w1, w2, b):
    X1, X2 = X[:, 0], X[:, 1]
    z = w1 * X1 + w2 * X2 + b
    # Ограничиваем y_pred через np.clip, чтобы избежать логарифма нуля log(0)
    y_pred = 1 / (1 + np.exp(-z))
    y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)
    return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

def get_gradients(X, y_true, w1, w2, b):
    X1, X2 = X[:, 0], X[:, 1]
    z = w1 * X1 + w2 * X2 + b
    y_pred = 1 / (1 + np.exp(-z))
    error_diff = y_pred - y_true

    # Для логистической регрессии (Log-Loss) коэффициента 2 НЕТ
    grad_w1 = np.mean(X1 * error_diff)
    grad_w2 = np.mean(X2 * error_diff)
    grad_b = np.mean(error_diff)

    return grad_w1, grad_w2, grad_b

def gradient_descent(X, y_true, learning_rate, steps, tolerance=1e-9):
    w1, w2, b = 0.0, 0.0, 0.0
    history = []
    for step in range(steps):
        grad_w1, grad_w2, grad_b = get_gradients(X, y_true, w1, w2, b)

        w1_new = w1 - grad_w1 * learning_rate
        w2_new = w2 - grad_w2 * learning_rate
        b_new = b - grad_b * learning_rate

        current_loss = calculate_log_loss(X, y_true, w1_new, w2_new, b_new)
        history.append(current_loss)

        if step % 2000 == 0 or step == steps - 1:
            print(f"Шаг {step:5d} | w1 = {w1:.3f}, w2 = {w2:.3f}, b = {b:.3f} | LogLoss = {current_loss:.5f}")

        if (abs(w1 - w1_new) < tolerance) and (abs(w2 - w2_new) < tolerance) and (abs(b - b_new) < tolerance):
            print("Алгоритм сошелся на шаге:", step)
            w1, w2, b = w1_new, w2_new, b_new
            break

        w1, w2, b = w1_new, w2_new, b_new

    return w1, w2, b, history

learning_rate = 0.1
steps = 20000
w1, w2, b, error_history = gradient_descent(X_train, y_train, learning_rate, steps)

y_train_pred = predict(X_train, w1, w2, b)
y_test_pred = predict(X_test, w1, w2, b)

train_loss = calculate_log_loss(X_train, y_train, w1, w2, b)
test_loss = calculate_log_loss(X_test, y_test, w1, w2, b)
print("\n--- Найденные коэффициенты ---")
print(f"w1 (рост) = {w1:.4f}, w2 (вес) = {w2:.4f}, b = {b:.4f}")
print("--- Оценка качества модели ---")
print(f"Log-Loss на Train: {train_loss:.4f}")
print(f"Log-Loss на Test:  {test_loss:.4f}")
print(f"Train Accuracy: {accuracy(y_train, y_train_pred) * 100:.2f}%")
print(f"Test Accuracy: {accuracy(y_test, y_test_pred) * 100:.2f}%")


plt.figure(figsize=(10, 6))

# Разделяем точки по реальным классам (y=0 и y=1), чтобы увидеть плоскость признаков
plt.scatter(X_train[y_train == 0, 0], X_train[y_train == 0, 1], color='blue', label='Класс 0 (Train)', alpha=0.5)
plt.scatter(X_train[y_train == 1, 0], X_train[y_train == 1, 1], color='orange', label='Класс 1 (Train)', alpha=0.5)
plt.scatter(X_test[y_test == 0, 0], X_test[y_test == 0, 1], color='blue', marker='s', s=60, label='Класс 0 (Test)', alpha=0.9)
plt.scatter(X_test[y_test == 1, 0], X_test[y_test == 1, 1], color='orange', marker='s', s=60, label='Класс 1 (Test)', alpha=0.9)

# Выводим разделяющую прямую (Decision Boundary: w1*x1 + w2*x2 + b = 0  =>  x2 = (-w1*x1 - b) / w2)
X1_line = np.linspace(0, 1, 100)
X2_line = (-w1 * X1_line - b) / w2
plt.plot(X1_line, X2_line, color='red', linewidth=3, label='Разделяющая граница модели')

plt.xlim(-0.05, 1.05)
plt.ylim(-0.05, 1.05)
plt.title('Логистическая регрессия: разделение классов (Рост и Вес)')
plt.xlabel('Нормализованный Рост (X1)')
plt.ylabel('Нормализованный Вес (X2)')
plt.legend()
plt.grid(True)
plt.show()