#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 27 15:04:49 2018

@author: Stefano Toniolo
"""

from .variable import Variable
import numpy as np

class DiscreteVariable(Variable):
    
    def __init__(self, name, domain):
        bounds = np.array([min(domain), max(domain)])
        self.norm_domain = [((i - min(domain))/(max(domain)-min(domain))) for i in domain]
        super(DiscreteVariable, self).__init__(name, domain, bounds, 'discrete', 1)
    
    
    def get_rounded(self, x, norm=False):
        '''
        Return the closest discrete value from the bounds given an input
        '''
        _x = np.zeros_like(x)
        for i, xval in enumerate(x):
            if norm:
                _x[i,0] = self.norm_domain[np.abs(self.norm_domain - xval).argmin()]
            else:
                _x[i,0] = self.domain[np.abs(self.domain - xval).argmin()]
        return _x