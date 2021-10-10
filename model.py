# Функция вычисления компонент вектора f для конкретной точки (х1, х2)
def f(x1, x2):
    #return [1, x1, x1**2, x2**2]
    return [1, x1, x1**2, x2**2, x2]


# Функция вичисления значения модели объекта, для конкретных
# точек плана и вектора параметров
def model(x1, x2, theta):
    f_x = f(x1, x2)
    #return theta[0] * f_x[0] + theta[1] * f_x[1] + theta[2] * f_x[2] + theta[3] * f_x[3]
    return theta[0] * f_x[0] + theta[1] * f_x[1] + theta[2] * f_x[2] + theta[3] * f_x[3] + theta[4] * f_x[4]
