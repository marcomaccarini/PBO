#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov  2 18:58:16 2018

@author: Stefano Toniolo
"""
import numpy as np
import math

def normalize_Y_f(Y, norm_type="stats"):
    if Y.shape[1] > 1:
        raise NotImplementedError("Multipe outputs not supported")
    else:
        if norm_type == "stats":
            Y_norm = Y - Y.mean()
            std = Y.std()
            if std > 0:
                Y_norm = Y_norm / std
        else:
            raise ValueError("Unknow normalizaion")
    return Y_norm


def normalize_X(domain, X_unnorm):
    (N, n) = np.shape(X_unnorm)
    X_norm = np.zeros((N, n))
    idx = 0
    for fvar in domain:
        if fvar.get_type() == 'categorical':
            dim = fvar.get_dim()
            X_norm[:, idx:idx + dim] = X_unnorm[:, idx:idx + dim]
            idx = idx + fvar.get_dim()
        else:
            col = X_unnorm[:, idx]
            bounds = fvar.get_bounds()
            X_norm[:, idx] = (col - bounds[0]) / (bounds[1] - bounds[0])
            idx = idx + 1
    return X_norm


def normalize_IDW(X_unnorm, max):
    col = X_unnorm
    bounds = [0, max]
    # X_norm = (col - bounds[0]) / (bounds[1] - bounds[0])
    X_norm = (bounds[1] - bounds[0]) * col + bounds[0]

    return X_norm


def scaleValues(x, maxval):
    # traforma valori in range (0..maxval)
    return maxval * (x - np.min(x)) / (np.ptp(x) + 0.00000000001)


def denormalize_X(domain, X_norm):
    # X_denorm = np.empty_like(X_norm)

    (N, n) = np.shape(X_norm)
    X_denorm = np.zeros((N, n))

    idx = 0
    for fvar in domain:
        if fvar.get_type() == 'categorical':
            dim = fvar.get_dim()
            X_denorm[:, idx:idx + dim] = X_norm[:, idx:idx + dim]
            idx = idx + fvar.get_dim()
        else:
            col = X_norm[:, idx]
            bounds = fvar.get_bounds()
            X_denorm[:, idx] = (bounds[1] - bounds[0]) * col + bounds[0]
            idx = idx + 1
    return X_denorm


def one_hot_to_cat(domain, X):
    new_X = np.empty((X.shape[0], 0))
    idx = 0
    for fvar in domain:
        if fvar.get_type() == 'categorical':
            dim = fvar.get_dim()
            new_X = np.column_stack((new_X, fvar.get_value(X[:, idx:idx + dim])))
            idx = idx + fvar.get_dim()
        else:
            new_X = np.column_stack((new_X, X[:, idx]))
            idx = idx + 1

    return new_X


def is_number(s):
    try:
        float(s)
        return not np.isnan(float(s))
    except ValueError:
        return False
