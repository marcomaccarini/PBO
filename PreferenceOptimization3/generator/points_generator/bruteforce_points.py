#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec  2 18:07:36 2018

@author: Stefano Toniolo
"""

from generator.points_generator.points_generator import PointsGenerator
import numpy as np


class BruteforcePointsGenerator(PointsGenerator):

    def __init__(self, fvars_not_cat, fvars_cat):
        super(BruteforcePointsGenerator, self).__init__()
        self.fvars_not_cat = fvars_not_cat
        self.fvars_cat = fvars_cat
        self.cat_rows, self.cat_cols = self._compute_dim(self.fvars_cat)

    def _compute_dim(self, fvars_cat):
        rows = 1
        cols = 0
        for fvar in fvars_cat:
            rows = rows * fvar.get_dim()
            cols = cols + fvar.get_dim()
        return rows, cols

    def generate_points(self, num_points):
        # Genrate random samples for non-categorical
        X_init_not_cat = np.empty((num_points, 0))

        for fvar in self.fvars_not_cat:
            if fvar.get_type() == 'continuous':
                x = np.random.uniform(fvar.get_bounds()[0], fvar.get_bounds()[1], (num_points, 1))
            elif fvar.get_type() == 'discrete':
                x = np.random.choice(fvar.get_domain(), (num_points, 1))
            else:
                raise ValueError('Variable type not found')
            X_init_not_cat = np.column_stack((X_init_not_cat, x))

        return X_init_not_cat

    def generate_points_cat_permutations(self):
        X_init_cat = np.empty((self.cat_rows, 0))
        X_init_cat_value = np.empty((self.cat_rows, 0))

        recursive_rows = self.cat_rows
        repeat = 1

        for i, fvar in enumerate(self.fvars_cat):
            domain = fvar.get_domain().reshape(-1, 1)
            one_hot = fvar.get_one_hot(domain)
            reps = int(recursive_rows / fvar.get_dim())

            columns = np.empty((0, fvar.get_dim()))
            columns_value = np.empty((0, 1))
            for i in range(reps):
                columns = np.row_stack((columns,
                                        np.repeat(one_hot, repeat, axis=0)))
                columns_value = np.row_stack((columns_value,
                                              np.repeat(domain, repeat, axis=0)))

            X_init_cat = np.column_stack((X_init_cat, columns))
            X_init_cat_value = np.column_stack((X_init_cat_value, columns_value))

            recursive_rows = reps
            repeat = repeat * fvar.get_dim()

        return X_init_cat, X_init_cat_value
