from .optimizer import Optimizer

import numpy as np
from scipy.optimize import minimize


class Powell(Optimizer):

    def __init__(self, bounds=None):
        super(Powell, self).__init__(bounds=bounds)

    def optimize(self, x0, f, maxiter, epsilon, df=None):

        res = minimize(f, x0=x0, bounds=self.bounds, method='Powell',
                       tol=epsilon, options={'maxiter': maxiter})

        result_x = np.atleast_2d(res.x)
        result_fx = np.atleast_2d(res.fun)

        return result_x, result_fx
