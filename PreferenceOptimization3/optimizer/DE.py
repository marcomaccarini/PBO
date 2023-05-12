#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 27 15:32:32 2018

@author: Stefano Toniolo
"""

from .optimizer import Optimizer
import numpy as np

from scipy.optimize import differential_evolution

class DE(Optimizer):

    def __init__(self, bounds=None):
        super(DE, self).__init__(bounds=bounds)

    def optimize(self, x0, f, maxiter, epsilon, df=None):
        res = differential_evolution(f, bounds=self.bounds, maxiter=maxiter, tol=epsilon)

        result_x = np.atleast_2d(res.x)
        result_fx = np.atleast_2d(res.fun)
        
        return result_x, result_fx
