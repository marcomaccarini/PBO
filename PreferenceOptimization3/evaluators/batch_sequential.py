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

        # x0 = np.array(x0, dtype=float)
        #
        #
        # x, y = self.acquisition_optimizer.optimize(
        #     x0,
        #     f=self.acquisition.get_acquisition,
        #     #f=self.acquisition,
        #     maxiter=maxiter,
        #     epsilon=epsilon
        # )
        x0 = np.array(x0, dtype=float)
        # If the domains are restricted by the user
        if restricted_domains_indexes is not None and restricted_domains_values is not None:
            restricted_domains_values = [x for _, x in
                                         sorted(zip(restricted_domains_indexes, restricted_domains_values))]
            restricted_domains_indexes = [x for x in sorted(restricted_domains_indexes)]
            bounds_prev = self.acquisition_optimizer.bounds.copy()
            discrete_restrictions = []
            discrete_idxs = []
            for count, idx in enumerate(restricted_domains_indexes):
                if self.vars[idx].get_type() == 'continuous':
                    if self.normalize_X:
                        b = self.vars[idx].get_bounds()
                        self.acquisition_optimizer.bounds[idx - len(discrete_idxs)] = (restricted_domains_values[
                                                                                           count] - b[0]) / (
                                                                                              b[1] - b[0])
                    else:
                        self.acquisition_optimizer.bounds[idx - len(discrete_idxs)] = restricted_domains_values[count]
                elif self.vars[idx].get_type() == 'discrete':
                    if self.normalize_X:
                        b = self.vars[idx].get_bounds()
                        discrete_restrictions.append((restricted_domains_values[count] - b[0]) / (b[1] - b[0]))
                    else:
                        discrete_restrictions.append(restricted_domains_values[count])
                    x0 = np.delete(x0, idx - len(discrete_idxs))
                    self.acquisition_optimizer.bounds = np.delete(self.acquisition_optimizer.bounds,
                                                                  idx - len(discrete_idxs), axis=0)
                    discrete_idxs.append(idx)
                else:
                    raise ValueError('Variable type not supported')

            # Use the following wrapper methods to iterate on restricted discrete variables' domains only if there
            # were discrete variable restrictions
            if len(discrete_idxs) != 0:

                grid = np.array(np.meshgrid(*discrete_restrictions)).T.reshape(-1, len(discrete_restrictions))

                def wrapper_discrete(x):
                    points = np.array([x for _ in range(len(grid))])
                    for count, idx in enumerate(discrete_idxs):
                        points = np.insert(points, idx, grid[:, count], axis=1)
                    G = self.acquisition.get_acquisition(points)
                    return min(G)

                def wrapper_discrete_grad(x):
                    points = np.array([x for _ in range(len(grid))])
                    for count, idx in enumerate(discrete_idxs):
                        points = np.insert(points, idx, grid[:, count], axis=1)
                    G = self.acquisition.get_acquisition_grad(points)
                    sel = G[np.argmin(np.min(G))]
                    for count, idx in enumerate(discrete_idxs):
                        sel = np.delete(sel, idx - count)
                    return sel

                x = np.array([[]])
                if len(x0) != 0:
                    if self.acquisition.grad_support():
                        x, y = self.acquisition_optimizer.optimize(
                            x0,
                            f=wrapper_discrete,
                            df=wrapper_discrete_grad,
                            maxiter=maxiter,
                            epsilon=epsilon
                        )
                    else:
                        x, y = self.acquisition_optimizer.optimize(
                            x0,
                            f=wrapper_discrete,
                            maxiter=maxiter,
                            epsilon=epsilon
                        )

                points = np.array([x[0] for _ in range(len(grid))])
                for count, idx in enumerate(discrete_idxs):
                    points = np.insert(points, idx, grid[:, count], axis=1)

                G = self.acquisition.get_acquisition(points)
                best_discrete = grid[np.argmin(G)]
                for count, idx in enumerate(discrete_idxs):
                    x = np.insert(x, idx, best_discrete[count], axis=1)

            # Otherwise perform a normal optimization (possibly limited on the continuous variables' bounds)
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
            self.acquisition_optimizer.bounds = bounds_prev

        # Otherwise perform a normal optimization
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
