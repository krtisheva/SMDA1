import numpy.linalg as npl
import scipy.stats
import pandas as pd
from gen_data import *


def lab1_2():
    answer = input('Хотите сгенерировать новые данные? (y/n)\n')
    if answer == 'y':
        x1, x2, etta, y, variation = data_gen()
        output_data(x1, x2, y)
    n, x1, x2, y = input_data()
    theta, inv_x, x = lsm(n, x1, x2, y)
    m = len(theta)
    # Вычисление несмещенной оценки неизвестной дисперсии
    dis = dispersion(y, x, theta, n, m)

    # Проверка гипотезы об адекватности модели
    print(check_hypothesis(n, m, dis))


def lab3():
    answer = input('Хотите сгенерировать новые данные? (y/n)\n')
    if answer == 'y':
        x1, x2, etta, y, variation = data_gen()
        output_data(x1, x2, y)
    n, x1, x2, y = input_data()
    theta, inv_x, x = lsm(n, x1, x2, y)
    m = len(theta)

    # Вычисление несмещенной оценки неизвестной дисперсии
    dis = dispersion(y, x, theta, n, m)

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


def lab4():
    answer = input('Хотите сгенерировать новые данные? (y/n)\n')
    if answer == 'y':
        x1, x2, y = data_gen()
        output_data(x1, x2, y)

    n, x1, x2, y = input_data()
    theta, inv_x, x = lsm(n, x1, x2, y)

    answer = input('Проверить данные на гетероскедастичность? (y/n)\n')
    if answer == 'y':
        theta_aux = breush_pagan_test(x1, x2, theta, y, x, n)
        goldfeld_quandt_test(x1, x2, y)

    v = np.zeros(shape=(n, n))
    for i in range(n):
        v[i, i] = 1 / (theta_aux[0] + theta_aux[1] * x1[i] ** 2 + theta_aux[2] * x2[i] ** 2)

    theta1 = glsm(x, v, y)
    theta_real = np.array([2.5, 2, 0.02, 1.2])
    diff0 = np.transpose(theta_real - theta) @ (theta_real - theta)
    diff1 = np.transpose(theta_real - theta1) @ (theta_real - theta1)
    print(f'Сумма квадратов расстояний МНК-оценки от истинного значения: {diff0}')
    print(f'Сумма квадратов расстояний ОМНК-оценки от истинного значения: {diff1}')


# Функция проверки гипотезы о гомоскедастичности по тесту Бройша-Пагана
def breush_pagan_test(x1, x2, theta, y, x, n):
    var_est = 0
    y_est = x @ theta                           # Предсказанные значения отклика
    e_t = y - y_est                             # Остаточная вариация

    # Цикл вычисления оценки дисперсии
    for i in range(0, n):
        var_est += e_t[i] ** 2
    var_est /= n
    print(f'Оценка дисперсии: {var_est}')

    c = np.zeros(n)                             # Отклик для вспомогательной регрессии
    for i in range(n):
        c[i] = e_t[i] ** 2 / var_est            # Вычисление отклика

    x_aux = np.zeros(shape=(n, 3))              # Матрица наблюдений для вспомогательной регрессии
    for i in range(n):
        x_aux[i] = z(x1[i], x2[i])              # Заполнение матрицы наблюдений

    theta_aux, x_inv = estimate(c, x_aux)       # Оценивание параметров вспомогательной модели
    print("МНК-оценка оценка параметров вспомогательной регрессии:\ntheta_aux = %s" % theta_aux)
    c_est = x_aux @ theta_aux                   # Предсказанный отклик вспомогательной модели
    c_avg = np.sum(c) / n                       # Среднее значение отклика всп. модели
    ess = 0.0                                   # Объясненная вариация

    for i in range(n):
        ess += (c_est[i] - c_avg) ** 2          # Вычисление объясненной вариации

    print("Проверка гипотезы о гомоскедастичности(Тест Бройша-Пагана):")
    chi2_dist = scipy.stats.chi2.ppf(q=1-0.05, df=1)
    if ess / 2 < chi2_dist:
        print('Возмущения гомоскедастичны  (%f < %f)' % (ess / 2, chi2_dist))
    else:
        print('Присутствует гетероскедастичность (%f >= %f)' % (ess / 2, chi2_dist))
    return theta_aux


# Функция проверки гипотезы о гомоскедастичности по тесту Гольдфельда-Квандта
def goldfeld_quandt_test(x1, x2, y):
    d = {'x1': x1, 'x2': x2, 'sq': [x1[i] ** 2 + x2[i] ** 2 for i in range(len(y))], 'y': y}
    df = pd.DataFrame(data=d)
    df.sort_values(by=['sq'], inplace=True)
    df.drop(range(75, 150, 1))

    df1 = df.iloc[:75]
    df2 = df.iloc[150:]

    x1_1 = df1.iloc[:, 0].values
    x2_1 = df1.iloc[:, 1].values
    y1 = df1.iloc[:, 3].values
    x_obs_1 = fill_obs_matrix(75, x1_1, x2_1)

    theta1, inv_x_1 = estimate(y1, x_obs_1)
    e1 = y1 - x_obs_1 @ theta1
    rss1 = np.transpose(e1) @ e1

    x1_2 = df2.iloc[:, 0].values
    x2_2 = df2.iloc[:, 1].values
    y2 = df2.iloc[:, 3].values
    x_obs_2 = fill_obs_matrix(75, x1_2, x2_2)

    theta2, inv_x_2 = estimate(y2, x_obs_2)
    e2 = y2 - x_obs_2 @ theta2
    rss2 = np.transpose(e2) @ e2

    print("Проверка гипотезы о гомоскедастичности(тест Гольдфельда-Куандта):")
    fisher_dist = scipy.stats.f.ppf(q=1 - 0.05, dfn=71, dfd=71)
    if rss2 / rss1 < fisher_dist:
        print('Возмущения гомоскедастичны  (%f < %f)' % (rss2 / rss1, fisher_dist))
    else:
        print('Присутствует гетероскедастичность (%f >= %f)' % (rss2 / rss1, fisher_dist))


# подпрограмма вычисления МНК-оценок параметров модели и
# проверки гипотезы об адекватности модели
def lsm(n, x1, x2, y):
    # Ввод информации о выборке наблюдений
    y = np.array(y)
    # Заполнение матрицы наблюдений X
    x = fill_obs_matrix(n, x1, x2)

    # Оценивание параметров модели объекта
    theta, inv_x = estimate(y, x)
    # m - размерность вектора неизвестных параметров
    m = len(theta)
    print("МНК-оценка оценка параметров модели объекта:\ntheta = %s" % theta)
    return theta, inv_x, x

# Функция вычисления оценок по ОМНК
def glsm(x, inv_v, y):
    x_t = np.transpose(x)
    #inv_v = npl.pinv(v)
    theta = npl.inv(x_t @ inv_v @ x) @ x_t @ inv_v @ y  # (X_t * V^-1 * X)^-1 X_t * V^-1 * y
    print("ОМНК-оценка оценка параметров модели объекта:\ntheta = %s" % theta)
    return theta


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
    fisher_dist = scipy.stats.f.ppf(q=1-0.05, dfn=1, dfd=n-m) # Квантиль распределения Фишера
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
