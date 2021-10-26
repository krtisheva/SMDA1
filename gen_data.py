import random
import math
import matplotlib.pyplot as plt
from model import *


# Подпрограмма генерации экспериментальных данных
def data_gen():
    etta = []   # вектор незашумленного отклика
    y = []      # вектор зашумленного отклика
    var_err = []  # вектор дисперсий ошибок измерения
    x1 = []     # вектор значений первого фактора
    x2 = []     # вектор значений второго фактора
    theta = [2.5, 2, 0.02, 1.2]     # истинные значения оцениваемых параметров
    # заполнение векторов etta, x1, x2
    for i in list(a / 7 for a in range(-7, 8)):
        for j in list(a / 7 for a in range(-7, 8)):
            etta.append(model(i, j, theta))
            x1.append(i)
            x2.append(j)

    print("Построение графика зависимости незашумленного отклика от входных факторов")
    # построение графика зависимости незашумленного отклика от факторов
    plotting(x1, x2, theta)
    n = len(etta)           # количество измерений
    avg = sum(etta) / n     # среднее значение незашумленного отклика
    # вычисление мощности сигнала
    omega2 = sum(list((etta[i] - avg) ** 2 for i in range(0, n))) / (n - 1)

    answer = input('Сгенерировать гетероскедастичное возмущение? (y/n)\n')
    # генерация шума и заполнение вектора зашумленного сигнала
    if answer == 'y':
        for i in range(0, n):
            variation = 0.1 * omega2 + x1[i] ** 2 + x2[i] ** 2
            e = random.normalvariate(0, math.sqrt(variation))
            y.append(etta[i] + e)
            var_err.append(variation)
    else:
        for i in range(0, n):
            variation = 0.1 * omega2
            e = random.normalvariate(0, math.sqrt(variation))
            y.append(etta[i] + e)
            var_err.append(variation)

    print("Построение графика зависимости дисперсии ошибки измерения от незашумленного отклика")
    var_plotting([x1[i] ** 2 + x2[i] ** 2 for i in range(n)], var_err)

    return x1, x2, y

# Построение графика зависимости дисперсии от отклика
def var_plotting(x1x2, var_err):
    plt.Figure()
    plt.suptitle("График зависимости дисперсии ошибки")
    plt.title("измерения от суммы квадратов факторов")
    plt.plot(x1x2, var_err)
    plt.xlabel("x1^2 + x2^2")
    plt.ylabel("error variation")
    plt.show()


# Функция вывода данных в файл унифицированной структуры
def output_data(x1, x2, y):
    # Открытие файла
    f_out = open("modelled_data.txt", "w")
    n = len(x1)     # число измерений
    f_out.write("%d\n" % n)
    for i in range(0, n):
        f_out.write("%f\t%f\t%f\n" % (x1[i], x2[i], y[i]))
    f_out.close()
    print("Сгенерированные данные вывели в файл 'modelled_data.txt'!")


# Функция построение графика зависимости незашумленного отклика от входных факторов
def plotting(x1, x2, theta):
    plt.figure()
    ax = plt.axes(projection='3d')
    ax.set_title("График зависимости незашумленного отклика от входных факторов")

    # Набор значений для построения поверхности
    x = tuple(x1)
    y = tuple(x2)
    z = tuple(list(model(x1[i], x2[i], theta) for i in range(0, len(x1))))

    # Установка названий осей
    ax.set_xlabel("X1")
    ax.set_ylabel("X2")
    ax.set_zlabel("etta")
    # Построение поверхности
    ax.plot_trisurf(x, y, z)
    ax.view_init(15, 140)
    plt.show()
