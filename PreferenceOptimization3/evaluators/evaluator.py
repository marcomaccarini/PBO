# -*- coding: utf-8 -*-

import numpy as np

from abc import ABC, abstractmethod


class Evaluator(ABC):
    """
    Base class for the evaluator of the function. This class handles both sequential and batch evaluators.
    """

    def __init__(self, acquisition, batch_size, vars,
                 normalize_X,
                 one_hot_permutations, cat_permutations, acquisition_optimizer):
        self.acquisition = acquisition
        self.batch_size = batch_size
        self.vars = vars
        self.normalize_X = normalize_X
        self.one_hot_permutations = one_hot_permutations
        self.cat_permutations = cat_permutations
        self.acquisition_optimizer = acquisition_optimizer

    @abstractmethod
    def compute_batch(self, anchor_points, maxiter, epsilon, duplicate_manager=None, restricted_domains_indexes=None, restricted_domains_values=None):
        pass

    # @abstractmethod
    # def _get_anchor_points(self):
    #     pass

    @abstractmethod
    def _optimize_anchor_point(self, x0, maxiter, epsilon, restricted_domains_indexes=None, restricted_domains_values=None):
        pass

    # Bruteforce wrapper functions
    def _bruteforce_acqu(self, anchor_point):
        G = []
        for perm in self.one_hot_permutations:
            G.append(self.acquisition.get_acquisition(np.hstack((anchor_point, perm))))
        return min(G)

    def _bruteforce_acqu_grad(self, anchor_point):
        G = []
        for perm in self.one_hot_permutations:
            G.append(self.acquisition.get_acquisition_grad(np.hstack((anchor_point, perm))))
        return min(G)

    def _get_rounded(self, x):
        x_round = np.zeros_like(x)
        idx = 0
        for fvar in self.vars:
            dim = fvar.get_dim()
            x_round[:, idx:idx + dim] = fvar.get_rounded(x[:, idx:idx + dim], self.normalize_X)
            idx = idx + dim
        return x_round
