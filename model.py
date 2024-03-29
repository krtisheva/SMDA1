import numpy as np


# Функция вычисления компонент вектора f для конкретной точки (х1, х2)
def f(x1, x2):
    return [1, x1, x1**2, x2**2]


# Функция вичисления значения модели объекта, для конкретных
# точек плана и вектора параметров
def model(x1, x2, theta):
    f_x = f(x1, x2)
    return theta[0] * f_x[0] + theta[1] * f_x[1] + theta[2] * f_x[2] + theta[3] * f_x[3]

# Функция вычисления компонент вектора z для конкретной точки плана
def z(x1, x2):
    return [1, x1 ** 2, x2 ** 2]


