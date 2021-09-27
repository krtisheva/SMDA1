import numpy as np
import numpy.linalg as npl
import scipy.stats
from gen_data import f


def input_data():
    with open('modelled_data.txt', 'r') as f_in:
        n = int(f_in.readline())
        x1 = []
        x2 = []
        y = []
        for line in f_in:
            s = line.split()
            x1.append(float(s[0]))
            x2.append(float(s[1]))
            y.append(float(s[2]))
    return n, x1, x2, y


def fill_obs_matrix(n, x1, x2):
    x = []
    for i in range(0, n):
        f_x = f(x1[i], x2[i])
        x.append(f_x)
    x = np.array(x)
    return x


def estimate(y, x):
    x_t = np.transpose(x)
    inv_x = npl.inv(np.matmul(x_t, x))
    theta = np.matmul(np.matmul(inv_x, x_t), y)
    return theta


def dispersion(y, x, theta, n, m):
    y_ =  np.matmul(x, theta)
    print("y^ = %s" % y_)
    e = y - y_
    print("y - y^ = %s" % e)
    dis = np.matmul(np.transpose(e), e) / (n - m)
    print("dis = %f" % dis)
    return dis


def check_hypothesis(n, m, dis):
    fisher_dist = scipy.stats.f.ppf(q=1-0.05, dfn=n-m, dfd=1000000)
    dis_e = 0.23459062500000002
    print("dis_e = %f" % dis_e)
    print("Табличное значение квантили F-распределения (Ft): %f" % fisher_dist)
    print("Статистика F: %f" % (dis / dis_e))
    return 'Гипотеза не отвергается (F <= Ft)' if dis / dis_e <= fisher_dist else 'Модель неадекватна (F > Ft)'
