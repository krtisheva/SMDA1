import random
import math
import matplotlib.pyplot as plt
from model import *


# Подпрограмма генерации экспериментальных данных
def data_gen():
    etta = []   # вектор незашумленного отклика
    y = []      # вектор зашумленного отклика
    x1 = []     # вектор значений первого фактора
    x2 = []     # вектор значений второго фактора
    theta = [2.5, 2, 0.02, 1.2]     # истинные значения оцениваемых параметров
    # заполнение векторов etta, x1, x2
    for i in list(a / 2 for a in range(-2, 3)):
        for j in list(a / 2 for a in range(-2, 3)):
            etta.append(model(i, j, theta))
            x1.append(i)
            x2.append(j)

    print("Построение графика зависимости незашумленного отклика от входных факторов")
    # построение графика зависимости незашумленного отклика от факторов
    plotting(x1, x2, theta)
    n = len(etta)           # количество измерений
    avg = sum(etta) / n     # стреднее значение незашумленного отклика
    # вычисление мощности сигнала
    omega2 = sum(list((etta[i] - avg) ** 2 for i in range(0, n))) / (n - 1)

    # генерация шума и заполнение вектора зашумленного сигнала
    for i in range(0, n):
        #e = random.normalvariate(0, math.sqrt(0.1 * omega2))
        e = random.normalvariate(0, math.sqrt(0.6 * omega2))
        y.append(etta[i] + e)

    return x1, x2, etta, y


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
