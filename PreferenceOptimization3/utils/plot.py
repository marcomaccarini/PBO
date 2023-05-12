#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 30 11:59:30 2018

@author: Stefano Toniolo
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
from utils.math import denormalize_X


def plot_approximation(gpr, X, Y, X_next, domain, normalize_X, constraints, grid=400, show_legend=False):
    if X_next.shape[1] == 1:
        if normalize_X:
            X_linspace = np.reshape(np.linspace(0, 1, 1000), (-1, 1))
        else:
            X_linspace = np.reshape(np.linspace(domain[0].bounds[0], domain[0].bounds[1], 1000), (-1, 1))

        mu, variance = gpr.predict(X_linspace)
        std = np.sqrt(variance)

        if normalize_X:
            X_linspace = denormalize_X(domain, X_linspace)

        plt.fill_between(X_linspace.ravel(),
                         mu.ravel() + 1.96 * std.ravel(),
                         mu.ravel() - 1.96 * std.ravel(),
                         alpha=0.1)
        plt.plot(X_linspace, mu, 'b-', lw=1, label='Surrogate function')
        plt.plot(X, Y, 'ro', mew=3, label='Samples')
        for x_to_draw in X_next:
            plt.axvline(x=x_to_draw, ls='--', c='r', lw=1)

        plt.xlabel('x')
        plt.ylabel('f(x)')
        plt.title('Surrogate function')

    elif X_next.shape[1] == 2:

        if normalize_X:
            X1 = np.linspace(0, 1, grid)
            X2 = np.linspace(0, 1, grid)
        else:
            X1 = np.reshape(np.linspace(domain[0].bounds[0], domain[0].bounds[1], grid), (-1, 1))
            X2 = np.reshape(np.linspace(domain[1].bounds[0], domain[1].bounds[1], grid), (-1, 1))

        x1, x2 = np.meshgrid(X1, X2)
        X_linspace = np.hstack((x1.reshape(grid * grid, 1), x2.reshape(grid * grid, 1)))

        X_mask = np.ones((X_linspace.shape[0], 1))
        if constraints != None:
            for c in constraints:
                if c['type'] == 'ineq':
                    yvals = c['fun'](X_linspace)
                    mask = (yvals < 0) * 1
                    X_mask *= mask.reshape((X_linspace.shape[0], 1))

        X_mask = X_mask.reshape(grid, grid)

        mu, _ = gpr.predict(X_linspace)
        mu = mu.reshape(grid, grid)
        mu = np.ma.masked_where(X_mask == 0, mu)

        X_linspace_init = np.hstack((X1.reshape(-1, 1), X2.reshape(-1, 1)))

        if normalize_X:
            X_linspace_init = denormalize_X(domain, X_linspace_init)

        x1, x2 = np.meshgrid(X_linspace_init[:, 0], X_linspace_init[:, 1])
        plt.pcolormesh(x1, x2, mu.reshape(grid, grid), cmap='viridis', label='Surrogate function')
        plt.colorbar()
        plt.plot(X[:, 0], X[:, 1], 'r.', mew=3, label='Samples')
        for x_to_draw in X_next:
            plt.plot(x_to_draw[0], x_to_draw[1], 'wo', mew=3, label='X next')

        plt.xlabel('x1')
        plt.ylabel('x2')
        plt.title('Surrogate function')

    else:
        raise ValueError("Cannot plot surrogate with more than 3 dimensions")

    if show_legend:
        plt.legend()


def plot_acquisition(gpr, domain, bounds, acquisition, X_next, normalize_X, constraints, grid=400, show_legend=True):
    if X_next.shape[1] == 1:

        if normalize_X:
            X = np.linspace(0, 1, 1000).reshape(-1, 1)
        else:
            X = np.reshape(np.linspace(domain[0].bounds[0], domain[0].bounds[1], 1000), (-1, 1))

        acqu = -acquisition.get_acquisition(X)
        acqu[acqu <= 0] = 0

        if max(acqu) != 0.0:
            acqu = (acqu - min(acqu)) / (max(acqu) - min(acqu))

        # GP uses normalized Xs, get_acquisition uses GP. Ergo, these Xs must be normalized
        if normalize_X:
            X = denormalize_X(domain, X)

        plt.plot(X, acqu, 'r', lw=1, label='Acquisition function')
        for x_to_draw in X_next:
            plt.axvline(x=x_to_draw, ls='--', c='r', lw=1, label='Next sampling location')

        plt.xlabel('x')
        plt.ylabel('p(xmin)')
        plt.title('Acquisition function')


    elif X_next.shape[1] == 2:
        X1 = np.linspace(bounds[0][0], bounds[0][1], grid)
        X2 = np.linspace(bounds[1][0], bounds[1][1], grid)
        x1_denorm, x2_denorm = np.meshgrid(X1, X2)

        if normalize_X:
            X1 = np.linspace(0, 1, grid)
            X2 = np.linspace(0, 1, grid)

        x1, x2 = np.meshgrid(X1, X2)
        X = np.hstack((x1.reshape(grid * grid, 1), x2.reshape(grid * grid, 1)))
        acqu = -acquisition.get_acquisition(X)

        X_mask = np.ones((X.shape[0], 1))
        if constraints != None:
            for c in constraints:
                if c['type'] == 'ineq':
                    yvals = c['fun'](X)
                    mask = (yvals < 0) * 1
                    X_mask *= mask.reshape((X.shape[0], 1))
        X_mask = X_mask.reshape(grid, grid)

        acqu[acqu == -1] = 0
        if max(acqu) != 0:
            acqu_normalized = (acqu - min(acqu)) / (max(acqu) - min(acqu))
        else:
            acqu_normalized = acqu
        acqu_normalized = acqu_normalized.reshape((grid, grid))

        acqu_normalized = np.ma.masked_where(X_mask == 0, acqu_normalized)

        plt.contourf(x1_denorm, x2_denorm, acqu_normalized, grid, cmap='viridis')

        # cs = plt.contour(x1_denorm, x2_denorm, acqu_normalized, cmap='viridis')
        # plt.clabel(cs, inline=1, fontsize=10)

        plt.colorbar()
        plt.grid(False)

        for x_to_draw in X_next:
            plt.plot(x_to_draw[0], x_to_draw[1], 'wo', mew=3, label='X next')

        plt.xlabel('x1')
        plt.ylabel('x2')
        plt.title('Acquisition function')

    else:
        raise ValueError("Cannot plot acquisition with more than 3 dimensions")

    if show_legend:
        plt.legend()
