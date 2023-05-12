#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 27 15:29:46 2018

@author: Stefano Toniolo
"""

from .optimizer import Optimizer

import numpy as np
from scipy.optimize import minimize

class SLSQP(Optimizer):

    def __init__(self, bounds=None):
        super(SLSQP, self).__init__(bounds=bounds)
        
    def optimize(self, x0, f, maxiter, epsilon, df=None):
        if df is None:
            res = minimize(f, x0=x0, bounds=self.bounds, method='SLSQP',
                           tol=epsilon, options={'maxiter': maxiter})
        else:
            res = minimize(f, jac=df, x0=x0, bounds=self.bounds, method='SLSQP',
                           tol=epsilon, options={'maxiter': maxiter})
            
        result_x = np.atleast_2d(res.x)
        result_fx = np.atleast_2d(res.fun)
        
        return result_x, result_fx
