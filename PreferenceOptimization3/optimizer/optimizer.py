#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  6 15:27:05 2018

@author: Stefano Toniolo
"""

import time
import warnings

from abc import ABC, abstractmethod


class Optimizer(ABC):
    """
    Abstract class representing the basic structure each optimization class has to implement
    """

    def __init__(self, bounds=None):
        """
        Constructor
        :param bounds: Optimization bounds
        """
        self.bounds = bounds

    @abstractmethod
    def optimize(self, x0, f, maxiter, epsilon, df=None):
        """
        Methods that implement the optimization procedure
        :param x0: Points
        :param f: Function
        :param df: Function's derivative
        :return: Optimized point and it's derivative
        """
        raise NotImplementedError("Under development")

    def set_bounds(self, bounds):
        """
        Set the optimization bounds
        :param bounds: New bounds to be applied
        :return:
        """
        self.bounds = bounds

    def set_maxiter(self, maxiter):
        """
        Set the maximum number of iterations
        :param maxiter: Max number of iterations
        :return:
        """
        self.maxiter = maxiter

    def set_epsilon(self, epsilon):
        """
        Set the tolerance of the optimization
        :param epsilon: Tolerance value
        :return:
        """
        self.epsilon = epsilon


# Time based constraint

class TookTooLong(Warning):
    pass


class MinimizeStopper(object):
    def __init__(self, max_sec=None):
        self.max_sec = max_sec
        self.start = time.time()

    def __call__(self, xk=None):
        if self.max_sec:
            print(xk)
            elapsed = time.time() - self.start
            if elapsed > self.max_sec:
                warnings.warn("Terminating optimization: time limit reached",
                              TookTooLong)
            else:
                print(f"Elapsed: {elapsed:.3f} sec")
