import random
import math
from mpl_toolkits import mplot3d
import numpy as np
import matplotlib.pyplot as plt


def model(x1, x2, theta):
    f = [1, x1, x1**2, x2**2]
    return theta[0] * f[0] + theta[1] * f[1] + theta[2] * f[2] + theta[3] * f[3]


def data_gen():
    etta = []
    y = []
    x1 = []
    x2 = []
    theta = [2.5, 2, 0.02, 1.2]
    for i in list(a / 2 for a in range(-2, 3)):
        for j in list(a / 2 for a in range(-2, 3)):
            etta.append(model(i, j, theta))
            x1.append(i)
            x2.append(j)

    plotting(x1, x2, theta)
    n = len(etta)
    avg = sum(etta) / n
    omega2 = sum(list((etta[i] - avg) ** 2 for i in range(0, n))) / (n - 1)

    for i in range(0, n):
        e = random.normalvariate(0, math.sqrt(0.1 * omega2))
        y.append(etta[i] + e)

    return x1, x2, etta, y


def data_out(x1, x2, etta, y):
    f = open("modelled_data.txt", "w")
    n = len(x1)
    f.write("%d\n" % n)
    for i in range(0, n):
        f.write("%f\t%f\t%f\n" % (x1[i], x2[i], y[i]))
    f.close()


def plotting(x1, x2, theta):
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    ax.set_title("График зависимости незашумленного отклика от входных факторов")

    x, y = np.meshgrid(x1, x2)
    z = model(x, y, theta)
    print(x)
    print(y)
    print(z)
    ax.plot_wireframe(x, y, z)
    ax.view_init(15, 120)
    plt.show()


print("Генерация экспериментальных данных:")
x1, x2, etta, y = data_gen()
print("x1 = %s\nx2 = %s\ny = %s\netta = %s" % (x1, x2, y, etta))
data_out(x1, x2, etta, y)

