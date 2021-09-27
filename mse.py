import numpy as np
import numpy.linalg as npl
import scipy.stats
from model import f


# подпрограмма вычисления МНК-оценок параметров модели и
# проверки гипотезы об адекватности модели
def mse():
    # Ввод информации о выборке наблюдений
    n, x1, x2, y = input_data()
    y = np.array(y)
    # Заполнение матрицы наблюдений X
    x = fill_obs_matrix(n, x1, x2)

    # Оценивание параметров модели объекта
    theta = np.array(estimate(y, x))
    # m - размерность вектора неизвестных параметров
    m = len(theta)
    print("МНК-оценка оценка параметров модели объекта:\ntheta = %s" % theta)

    # Вычисление несмещенной оценки неизвестной дисперсии
    dis = dispersion(y, x, theta, n, m)

    # Проверка гипотезы об адекватности модели
    print(check_hypothesis(n, m, dis))

# Функция ввода данных о выборке измерений
def input_data():
    with open('modelled_data.txt', 'r') as f_in:
        n = int(f_in.readline())    # n - число измерений
        x1 = []                     # x1 - вектор значений фактора х1
        x2 = []                     # x2 - вектор значений фактора х2
        y = []                      # y - вектор наблюдений
        # цикл по всем строкам файла и заполнение определенных выше массивов
        for line in f_in:
            s = line.split()
            x1.append(float(s[0]))
            x2.append(float(s[1]))
            y.append(float(s[2]))
    return n, x1, x2, y


# Функция заполнения матрицы наблюдений X
def fill_obs_matrix(n, x1, x2):
    x = []                      # x - матрица наблюдений
    # В цикле от 0 до n вычисляем компоненты вектора f
    # для каждого наблюдения и заносим в матрицу X, как
    # новую строку, итого n строк по m элементов (n*m)
    for i in range(0, n):
        f_x = f(x1[i], x2[i])
        x.append(f_x)
    x = np.array(x)
    return x


# Функция оценивания неизвестных параметров модели
def estimate(y, x):
    x_t = np.transpose(x)                           # Транспонирование X
    inv_x = npl.inv(np.matmul(x_t, x))              # Нахождение обратной матрицы от X_t * X
    # Умножение произведения обратной матрицы
    # и транспонированной на вектор наблюдений
    # theta = ((X_t*X)^(-1)*X_t)y
    theta = np.matmul(np.matmul(inv_x, x_t), y)
    return theta


# Функция вычисления несмещенной оценки неизвестной дисперсии
def dispersion(y, x, theta, n, m):
    y_ = np.matmul(x, theta)    # Умножение матрицы наблюдений на вектор оцененных параметров
    print("y^ = %s" % y_)
    e = y - y_                  # Вычитание вектора полученного выше из вектора наблюдений
    print("y - y^ = %s" % e)
    dis = np.matmul(np.transpose(e), e) / (n - m)   # Вычисление оценки неизв. дисперсии
    print("dis = %f" % dis)
    return dis


# Функция проверки гипотезы об адекватности модели
def check_hypothesis(n, m, dis):
    # Вычисление значения квантили F-распределения при a=0.05,
    # n-m степенях свободы для оценки неизв. дисперсии и
    # бесконечности (очень большое число) степенях свободы для sigma_E
    fisher_dist = scipy.stats.f.ppf(q=1-0.05, dfn=n-m, dfd=1000000)
    # Значение дисперсии, используемой для генерации шумов в 1 ЛР
    dis_e = 0.23459062500000002
    print("dis_e = %f" % dis_e)
    print("Табличное значение квантили F-распределения (Ft): %f" % fisher_dist)
    print("Статистика F: %f" % (dis / dis_e))
    # Проверка гипотезы об адекватности и возвращение соотв. значения
    return 'Гипотеза не отвергается (F <= Ft)' if dis / dis_e <= fisher_dist else 'Модель неадекватна (F > Ft)'
