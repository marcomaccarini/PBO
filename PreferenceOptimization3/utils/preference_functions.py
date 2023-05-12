# -*- coding: utf-8 -*-
"""
Created on Thu Nov 14 15:19:22 2019

@author: Install
"""

import numpy as np


def compute_preference(f, x1, x2, **kwargs):
    # print(kwargs['parameters'])
    # param = kwargs['parameters']
    f_1 = f(x1)
    f_2 = f(x2)

    # print("X1: ", x1, "f1:", f_1)
    # print("X2: ", x2, "f2:", f_2)

    N_val = np.shape(x1)[0]

    pref_hat = np.zeros(N_val)

    for ind in range(N_val):
        if f_1[ind] <= f_2[ind]:
            pref_hat[ind] = 1
        else:
            pref_hat[ind] = -1

    # print('Minimum of the cost so far;  Cost value at next query point ')
    # print(f_1, f_2)
    return pref_hat


def compute_class(f, x, **kwargs):
    if kwargs != None:
        if 'my_fun' in kwargs.keys():

            f_obj = kwargs['my_fun']
            return f(x, f_obj)

        else:
            return f(x)

    else:
        return f(x)
