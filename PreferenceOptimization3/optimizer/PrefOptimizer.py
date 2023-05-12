#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from .optimizer import Optimizer
from .optimizer import MinimizeStopper
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize
from utils.math import denormalize_X


class LBFGSB(Optimizer):

    def __init__(self, bounds=None):
        super(LBFGSB, self).__init__(bounds=bounds)

    def optimize(self, x0, f, maxiter, epsilon, df=None):

        if df is None:
            res = minimize(f, x0=x0, bounds=self.bounds, method='L-BFGS-B', tol=epsilon,
                           options={'disp': None, 'maxcor': 10, 'ftol': 1e-09, 'gtol': 1e-05, 'eps': 1e-08,
                                    'maxfun': 15000, 'maxiter': 15000, 'iprint': -1, 'maxls': 20})

        else:
            res = minimize(f, jac=df, x0=x0, bounds=self.bounds, method='L-BFGS-B', tol=epsilon,
                           options={'maxiter': maxiter})
        # print("**** res ****")
        # print(res)
        result_x = np.atleast_2d(res.x)
        result_fx = np.atleast_2d(res.fun)

        return result_x, result_fx

    def reorder(self, xd, a):
        ee2 = a.reshape(len(a), 1)
        fin2 = np.column_stack((xd, ee2))
        fin2 = fin2[fin2[:, 0].argsort(), :]
        return fin2[:, 0], fin2[:, 1]
