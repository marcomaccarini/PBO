#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec  2 17:01:17 2018

@author: Stefano Toniolo
"""

from PreferenceOptimization3.generator.points_generator.points_generator import PointsGenerator

import numpy as np


class RandomPointsGenerator(PointsGenerator):

    def __init__(self, fvars):
        super(RandomPointsGenerator, self).__init__()
        self.fvars = fvars

    def generate_points(self, num_points):
        X_init = np.empty((num_points, 0))
        for fvar in self.fvars:
            if fvar.get_type() == 'continuous':
                x = np.random.uniform(fvar.get_domain()[0], fvar.get_domain()[1], (num_points, 1))
            elif fvar.get_type() == 'discrete':
                x = np.random.choice(fvar.get_domain(), (num_points, 1))
            elif fvar.get_type() == 'categorical':
                x = fvar.get_one_hot(np.random.choice(fvar.get_domain(), (num_points, 1)))
            else:
                raise ValueError('Variable type not found')
            X_init = np.column_stack((X_init, x))
        return X_init
