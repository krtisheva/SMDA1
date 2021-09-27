from gen_data import *
from mse import *


answer = input('Хотите сгенерировать новые данные? (y/n)\n')
if answer == 'y':
    x1, x2, etta, y = data_gen()
    output_data(x1, x2, y)

answer = input('Хотите оценить параметры модели объекта? (y/n)\n')
if answer == 'y':
    mse()
