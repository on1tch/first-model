from arrays import array_sum, array_mean, array_min, array_max, array_shape
from metrics import mse, mae, accuracy
from preprocessing import normalize
from data import train_test_split
import matplotlib.pyplot as plt
import random
import numpy as np

# --- 1. ГЕНЕРАЦИЯ ДАННЫХ ---
# Создадим побольше точек для параболы (y = 2x^2 + x - 2 + шум)

random.seed(42)
num_points = 10000
X_all = []
y_all = []

for _ in range(num_points):
    x_i = random.uniform(-5.0, 5.0)
    # Истинная формула: y = 2x^2 + 1x - 2 + шум
    y_i = 2 * (x_i**2) + 1 * x_i - 2 + random.uniform(-1, 1)
    X_all.append(x_i)
    y_all.append(y_i)
X_train_list, X_test_list, y_train_list, y_test_list = train_test_split(X_all, y_all)
X_train = np.array(X_train_list)
X_test = np.array(X_test_list)
y_train = np.array(y_train_list)
y_test = np.array(y_test_list)
def calculate_mse(X, y_true, w1,w2, b):
    y_pred = w1*X**2 + w2*X + b
    return np.mean((y_pred - y_true) ** 2)

def get_gradients(X, y_true, w1,w2,b):
    y_pred = w1 * (X**2) + w2*X + b
    error_diff = y_pred - y_true
    grad_w1 = 2 * np.mean((X**2) * error_diff)
    grad_w2 = 2 * np.mean(X * error_diff)
    grad_b = 2* np.mean(error_diff)

    return grad_w1, grad_w2, grad_b

def gradient_descent(X, y_true, learning_rate, steps, tolerance=1e-9):
    w1 = 0.0
    w2 = 0.0
    b = 0.0
    history = []
    for step in range(steps):
        grad_w1, grad_w2, grad_b = get_gradients(X, y_true, w1, w2, b)

        w1_new = w1 - grad_w1 * learning_rate
        w2_new = w2 - grad_w2 * learning_rate
        b_new = b - grad_b * learning_rate

        current_mse = calculate_mse(X, y_true, w1_new, w2_new, b_new)
        history.append(current_mse)

        if step % 500 == 0 or step == steps - 1:
            print(f"Шаг {step:4d} | w1 = {w1:.3f}, w2 = {w2:.3f}, b = {b:.3f} | MSE = {current_mse:.5f}")

        if (abs(w1 - w1_new) < tolerance) and (abs(w2 - w2_new) < tolerance) and (abs(b - b_new) < tolerance):
            print("step:", step)
            w1, w2, b = w1_new, w2_new, b_new
            break

        w1, w2, b = w1_new, w2_new, b_new

    return w1, w2, b, history

learning_rate = 0.0001
steps = 100000
w1, w2, b, error_history = gradient_descent(X_train, y_train, learning_rate, steps)

train_mse = calculate_mse(X_train, y_train, w1, w2, b)
test_mse = calculate_mse(X_test, y_test, w1, w2, b)
print("\n--- Найденные коэффициенты ---")
print(f"w1 = {w1:.4f}, w2 = {w2:.4f}, b = {b:.4f}")
print("--- Оценка качества модели ---")
print(f"Ошибка на обучающих данных (Train MSE): {train_mse:.4f}")
print(f"Ошибка на новых данных (Test MSE):      {test_mse:.4f}")

plt.figure(figsize=(10, 6))

# Рисуем тренировочные и тестовые точки разными цветами
plt.scatter(X_train, y_train, color='blue', label='Обучающие данные (Train)', alpha=0.7)
plt.scatter(X_test, y_test, color='green', marker='s', s=80, label='Новые данные (Test)')

# Строим плавную линию предсказания модели
X_line = np.linspace(-5, 5, 100)
y_line = w1 * (X_line**2) + w2 * X_line + b
plt.plot(X_line, y_line, color='red', linewidth=2.5, label='Предсказание модели')


plt.title('Проверка модели на новых (зеленых) данных')
plt.xlabel('X')
plt.ylabel('y')
plt.legend()
plt.grid(True)
plt.show()