# -*- coding: utf-8 -*-

import numpy as np

from .evaluator import Evaluator
from generator.anchor_points_generator.thompson_anchor_points import ThompsonAnchorPointsGenerator


class ThompsonBatch(Evaluator):
    """
    Class for a Thompson batch method. Elements are selected iteratively using the current acquistion function but exploring the models
    by using Thompson sampling
    :param acquisition: acquisition function to be used to compute the batch.
    :param batch size: the number of elements in the batch.
    """

    def __init__(self, acquisition, batch_size, vars, normalize_X, one_hot_permutations, acquisition_optimizer):

        super(ThompsonBatch, self).__init__(acquisition, batch_size, vars, normalize_X,
                                            one_hot_permutations, acquisition_optimizer)

    def _optimize_anchor_point(self, x0, maxiter, epsilon, restricted_domains_indexes=None,
                               restricted_domains_values=None):

        if self.samples_generator_type == 'bruteforce':
            x, y = self.acquisition_optimizer.optimize(x0=x0, maxiter=maxiter, epsilon=epsilon, f=self._bruteforce_acqu)
            # search for optimal cat
            G = []
            for perm in self.one_hot_permutations:
                G.append(self.acquisition.get_acquisition(np.hstack((x[0], perm))))
            x = np.hstack((x, np.atleast_2d(self.one_hot_permutations[np.argmin(G), :])))
        else:
            if self.acquisition.grad_support():
                x, y = self.acquisition_optimizer.optimize(
                    x0,
                    f=self.acquisition.get_acquisition,
                    df=self.acquisition.get_acquisition_grad,
                    maxiter=maxiter,
                    epsilon=epsilon
                )
            else:
                x, y = self.acquisition_optimizer.optimize(
                    x0,
                    f=self.acquisition.get_acquisition,
                    maxiter=maxiter,
                    epsilon=epsilon
                )
        return self._get_rounded(x)

    def compute_batch(self, anchor_points, maxiter, epsilon, duplicate_manager=None, restricted_domains_indexes=None,
                      restricted_domains_values=None):
        # Do not care about having duplicates suggested
        if not duplicate_manager:
            return self.compute_batch_without_duplicate_logic(anchor_points=anchor_points,
                                                              restricted_domains_indexes=restricted_domains_indexes,
                                                              restricted_domains_values=restricted_domains_values,
                                                              maxiter=maxiter,
                                                              epsilon=epsilon)
        else:
            raise Exception("Can't handle duplicate Xs")

    def compute_batch_without_duplicate_logic(self, anchor_points, maxiter, epsilon, restricted_domains_indexes=None,
                                              restricted_domains_values=None):
        # anchor_points = self.anchor_points_generator.get()
        return np.vstack([self._optimize_anchor_point(a, maxiter=maxiter, epsilon=epsilon,
                                                      restricted_domains_indexes=restricted_domains_indexes,
                                                      restricted_domains_values=restricted_domains_values) for a, _ in
                          zip(anchor_points, range(self.batch_size))])
