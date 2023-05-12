#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec 22 16:12:03 2018
@author: Marco Maccarini
"""
from numpy import linalg as LA
from .acquisition import Acquisition
import numpy as np
from PreferenceOptimization3.utils.math import one_hot_to_cat, denormalize_X, normalize_X, scaleValues
# import matplotlib
# matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
from mpl_toolkits.mplot3d import Axes3D
import PreferenceOptimization3.utils
import math


class Preferences(Acquisition):

    def __init__(self, model, N=999, jitter=0.0):
        super(Preferences, self).__init__(model, False)
        self.N = N
        self.jitter = jitter

    def _improvement(self, gmin, X, y):
        return np.maximum(0, gmin - self.model.g(X, y, self.model) - self.jitter)

    def normalize(self, x):
        if len(x) == 1:
            return x
        min = np.min(x)
        max = np.max(x)
        return (x - min) / (max - min)

    def _compute_acquisition(self, x):
        (N, n) = np.shape(self.model.X)
        x = np.reshape(x, (-1, n))
        xd = denormalize_X(self.model.fvars_x, x)
        nx = np.shape(x)[0]

        f_hat_sum = np.zeros(len(xd))
        f_hat_X_sum = np.zeros(len(self.model.X))

        forPlot = []

        a1 = []
        a2 = []

        bestYY = []
        index = 0
        indiceModello = 0

        # x next calcolato attraverso la combinazione di due modelli
        for m in self.model.models:
            xYY = m.X[self.model.Y_ind_best[index]]
            # f_hat_X = np.add(f_hat_X, self.normalize(m.predict(m.X)))
            f_hat_X = m.predict(m.X)
            mm = np.min(f_hat_X)
            MM = np.max(f_hat_X)
            Delta_F = MM - mm
            f_hat = m.predict(xd[:, indiceModello * self.model.ratio:(indiceModello + 1) * self.model.ratio])
            YYY = (m.predict(xYY) - mm) / Delta_F
            bestYY.append(YYY)
            f_hat = (f_hat - mm) / Delta_F
            forPlot.append(f_hat)
            # f_hat_sum = np.add(f_hat_sum, f_hat)
            # f_hat_X_sum = np.add(f_hat_X_sum,  self.normalize(f_hat_X))
            a1.append(self.normalize(f_hat_X))
            a2.append(f_hat)
            index += 1
            indiceModello += 1

        f_hat_X_sum = self.model.g(self.model.X, np.array(a1).T, self.model)
        f_hat_sum = self.model.g(self.model.X, np.array(a2).T, self.model)

        mm = np.min(f_hat_X_sum)
        MM = np.max(f_hat_X_sum)
        Delta_F = MM - mm

        forPlot2 = []
        z = np.zeros(len(x))
        indiceModello2 = 0
        for m in self.model.models:
            z = np.add(z, m.IDW(x[:, indiceModello2 * self.model.ratio:(indiceModello2 + 1) * self.model.ratio]))
            forPlot2.append(m.IDW(x[:, indiceModello2 * self.model.ratio:(indiceModello2 + 1) * self.model.ratio]))
            indiceModello2 += 1

        self.delta = self.calculate_delta()
        forPlot3 = []
        indiceModello7 = 0
        for m in self.model.models:
            forPlot3.append(
                1 * (forPlot[indiceModello7] - mm) / Delta_F - 1 * self.delta * forPlot2[indiceModello7])
            indiceModello7 += 1

        a = 1 * (f_hat_sum - mm) / Delta_F - 1 * self.delta * z
        bestYY = (bestYY - mm) / Delta_F

        if self.model.init == 1:
            for i in range(len(self.model.X_batch) - 1):
                d = (LA.norm(x - self.model.X_batch[i, :]) ** 2) / (1 ** 2)
                a = a * (1 - np.exp(-d))
        return a

    def _compute_acquisition_grad(self, x):
        raise NotImplementedError('Gradient not available for this Acquisition function')

    def calculate_delta(self):
        if self.exploration:
            d = self.model.delta
        else:
            d = 0
        return d
