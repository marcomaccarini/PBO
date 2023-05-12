# -*- coding: utf-8 -*-

import numpy as np
import sys
from .evaluator import Evaluator


class BatchSequential(Evaluator):
    """
    This class handles both single X_next and multiple X_next with same logic
    (objective is acquisition function)
    """

    def __init__(self, acquisition, batch_size, vars, normalize_X, one_hot_permutations=None, cat_permutations=None,
                 acquisition_optimizer=None):

        super(BatchSequential, self).__init__(acquisition, batch_size, vars, normalize_X,
                                              one_hot_permutations, cat_permutations, acquisition_optimizer)

    def _optimize_anchor_point(self, x0, maxiter, epsilon, restricted_domains_indexes=None,
                               restricted_domains_values=None):

        x0 = np.array(x0, dtype=float)

        x, y = self.acquisition_optimizer.optimize(
            x0,
            f=self.acquisition.get_acquisition,
            # f=self.acquisition,
            maxiter=maxiter,
            epsilon=epsilon
        )

        return self._get_rounded(x)

    def compute_batch(self, anchor_points, maxiter, epsilon, duplicate_manager=None, restricted_domains_indexes=None,
                      restricted_domains_values=None):
        # Do not care about having duplicates suggested
        if not duplicate_manager:
            return self.compute_batch_without_duplicate_logic(anchor_points, maxiter, epsilon,
                                                              restricted_domains_indexes, restricted_domains_values)
        else:
            raise Exception("Can't handle duplicate Xs")

    def compute_batch_without_duplicate_logic(self, anchor_points, maxiter, epsilon, restricted_domains_indexes=None,
                                              restricted_domains_values=None):
        # anchor_points = self._get_anchor_points()
        if self.batch_size == 1:
            # Optimize each anchor point
            optimized_x = np.zeros_like(anchor_points)
            for i, a in enumerate(anchor_points):
                optimized_x[i] = self._optimize_anchor_point(a, maxiter=maxiter, epsilon=epsilon,
                                                             restricted_domains_indexes=restricted_domains_indexes,
                                                             restricted_domains_values=restricted_domains_values)

            # Evaluate the anchor point
            optimized_ys = self.acquisition._compute_acquisition(optimized_x)
            return np.atleast_2d(optimized_x[np.argmin(optimized_ys)])
        else:
            raise NotImplementedError('Batch size != 1 not currently supported')
