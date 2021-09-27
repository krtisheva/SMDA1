from gen_data import *
from mse import *

"""print("Генерация экспериментальных данных:")
x1, x2, etta, y = data_gen()
print("x1 = %s\nx2 = %s\ny = %s\netta = %s" % (np.array(x1), np.array(x2), np.array(y), np.array(etta)))
output_data(x1, x2, y)"""

n, x1, x2, y = input_data()
y = np.array(y)
x = fill_obs_matrix(n, x1, x2)

theta = np.array(estimate(y, x))
m = len(theta)
print("МНК-оценка оценка параметров модели объекта:\ntheta = %s" % theta)

dis = dispersion(y, x, theta, n, m)
print(check_hypothesis(n, m, dis))
