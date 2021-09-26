from gen_data import *
from mse import *


print("Генерация экспериментальных данных:")
x1, x2, etta, y = data_gen()
print("x1 = %s\nx2 = %s\ny = %s\netta = %s" % (x1, x2, y, etta))
output_data(x1, x2, y)
n, x1, x2, etta = input_data()
etta = np.array(etta)

x = fill_obs_matrix(n, x1, x2)
theta = np.array(estimate(etta, x))
dis = dispersion(etta, x, theta, n, 4)
print(theta)
print(dis)

print(check_hypothesis(25, 4, dis))
