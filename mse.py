import math
import matplotlib.pyplot as plt
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
    theta, inv_x = estimate(y, x)
    # m - размерность вектора неизвестных параметров
    m = len(theta)
    print("МНК-оценка оценка параметров модели объекта:\ntheta = %s" % theta)

    # Вычисление несмещенной оценки неизвестной дисперсии
    dis = dispersion(y, x, theta, n, m)

    # Проверка гипотезы об адекватности модели
    print(check_hypothesis(n, m, dis))

    answer = input('Построить доверительные интервалы для каждого параметра модели регрессии? (y/n)\n')
    if answer == 'y':
        print("Доверительные интервалы для каждого параметра модели регрессии:")
        confidential_assessment_theta(n, m, dis, inv_x, theta)

    answer = input('Проверить гипотезу о незначимости каждого параметра модели? (y/n)\n')
    if answer == 'y':
        print("Проверка гипотезы о незначимости каждого параметра модели:")
        check_hypothesis_insignificance_parameters(n, m, dis, inv_x, theta)

    answer = input('Проверить гипотезу о незначимости самой регрессии? (y/n)\n')
    if answer == 'y':
        print("Проверка гипотезы о незначимости самой регрессии:")
        check_hypothesis_insignificance_regression(n, m, x, y, theta)

    answer = input('Рассчитать прогнозные значения для математического ожидания функции отклика? (y/n)\n')
    if answer == 'y':
        confidential_assessment_math_expectation(n, m, dis, inv_x, theta)


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
    return np.array(theta), inv_x


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
    fisher_dist = scipy.stats.f.ppf(q=1-0.05, dfn=n-m, dfd=10000000)
    # Значение дисперсии, используемой для генерации шумов в 1 ЛР
    #dis_e = 0.23459062500000002
    dis_e = 1.40754375
    print("dis_e = %f" % dis_e)
    print("Табличное значение квантили F-распределения (Ft): %f" % fisher_dist)
    print("Статистика F: %f" % (dis / dis_e))
    # Проверка гипотезы об адекватности и возвращение соотв. значения
    return 'Гипотеза не отвергается (F <= Ft)' if dis / dis_e <= fisher_dist else 'Модель неадекватна (F > Ft)'


# Функция построения доверительных интервалов для каждого параметра модели регрессии
def confidential_assessment_theta(n, m, dis, inv_x, theta):
    t = scipy.stats.t.ppf(q=1-0.05/2, df=n-m) # Квантиль распределения Стьюдента

    # Расчет левой и правой границ для каждого параметра
    for i in range(0, m):
        sigma = math.sqrt(dis * inv_x[i][i])
        left = theta[i] - t * sigma
        right = theta[i] + t * sigma
        print("%f <= theta%d <= %f" % (left, i+1, right))


# Функция роверки гипотезы о незначимости каждого параметра модели
def check_hypothesis_insignificance_parameters(n, m, dis, inv_x, theta):
    fisher_dist = scipy.stats.f.ppf(q=1-0.05, dfn=m-1, dfd=n-m) # Квантиль распределения Фишера
    for i in range(0, m):
        f = theta[i]**2 / (dis * inv_x[i][i])       # Статистика
        if f < fisher_dist:
            print('Параметр theta%d незначим (%f < %f)' % (i+1, f, fisher_dist))
        else:
            print('Параметр theta%d значим (%f >= %f)' % (i+1, f, fisher_dist))


# Функция проверки гипотезы о незначимости самой регрессии
def check_hypothesis_insignificance_regression(n, m, x, y, theta):
    y_avg = np.sum(y) / n
    rssh = 0

    for i in range(0, n):
        rssh += (y[i] - y_avg)**2

    div = y - np.matmul(x, theta)
    rss = np.matmul(np.transpose(div), div)
    f = ((rssh - rss) / (m - 1)) / (rss / (n - m))      # Статистика
    fisher_dist = scipy.stats.f.ppf(q=1-0.05, dfn=m-1, dfd=n-m)    # Квантиль распределения Фишера

    if f < fisher_dist:
        print('Регрессия незначима (%f < %f)' % (f, fisher_dist))
    else:
        print('Регрессия значима (%f >= %f)' % (f, fisher_dist))


# Функция рассчета прогнозных значений для математического ожидания функции отклика и самого отклика
def confidential_assessment_math_expectation(n, m, dis, inv_x, theta):
    x1 = 1
    x2 = np.zeros(5)
    left_math_exp_response = np.zeros(5)
    right_math_exp_response = np.zeros(5)
    left_response = np.zeros(5)
    right_response = np.zeros(5)

    t = scipy.stats.t.ppf(q=1 - 0.05 / 2, df=n - m) # Квантиль распределения Стьюдента

    print("Доверительное оценивание для математического ожидания отклика:")
    print("x1\tx2\t\t\tetta_left\tetta_right")
    for i in range(-2, 3):
        x2[i+2] = i / 2
        fun = f(x1, x2[i+2])
        # Дисперсия оценки математического ожидания функции отклика
        sigma_math_exp_response = math.sqrt(dis * np.matmul(np.matmul(np.transpose(fun), inv_x), fun))
        # Дисперсия оценки отклика
        sigma_response = math.sqrt(dis * (1 + np.matmul(np.matmul(np.transpose(fun), inv_x), fun)))
        etta = np.matmul(np.transpose(fun), theta)
        # Левая и правая границы доверительного интервала для оценки математического ожидания функции отклика
        left_math_exp_response[i+2] = etta - t * sigma_math_exp_response
        right_math_exp_response[i+2] = etta + t * sigma_math_exp_response
        # Левая и правая границы доверительного интервала для оценки отклика
        left_response[i+2] = etta - t * sigma_response
        right_response[i+2] = etta + t * sigma_response
        print("%d\t%f\t%f\t%f" % (x1, x2[i+2], left_math_exp_response[i+2], right_math_exp_response[i+2]))

    print("Доверительное оценивание для отклика:")
    print("x1\tx2\t\t\tetta_left\tetta_right")
    for i in range(0, 5):
        print("%d\t%f\t%f\t%f" % (x1, x2[i], left_response[i], right_response[i]))

    plotting_response(x1, x2, left_math_exp_response, right_math_exp_response, left_response, right_response, theta)


# Функция построения графиков прогнозных значений и доверительной полосы для математического ожидания функции отклика
# и для самого отклика
def plotting_response(x1, x2, left_math_exp_response, right_math_exp_response, left_response, right_response, theta):
    etta = np.zeros(5)
    for i in range(0, 5):
        etta[i] = np.matmul(np.transpose(f(x1, x2[i])), theta)

    plt.Figure()
    plt.suptitle("Прогнозные значения и доверительная полоса")
    plt.title("для математического ожидания функции отклика")
    plt.plot(x2, left_math_exp_response)
    plt.plot(x2, etta)
    plt.plot(x2, right_math_exp_response)
    plt.xlabel("X2")
    plt.ylabel("etta(1, x2, theta)")
    plt.show()

    plt.suptitle("Прогнозные значения и доверительная полоса")
    plt.title("для функции отклика")
    plt.plot(x2, left_response)
    plt.plot(x2, etta)
    plt.plot(x2, right_response)
    plt.xlabel("X2")
    plt.ylabel("y(1, x2, theta)")
    plt.show()
