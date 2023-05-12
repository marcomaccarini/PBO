#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 27 15:07:24 2018

@author: Stefano Toniolo
"""

from .variable import Variable
import numpy as np

class CategoricalVariable(Variable):
    
    def __init__(self, name, domain):
        bounds = np.empty((domain.shape[0], 2))
        for i in range(domain.shape[0]):
            bounds[i,:] = [0,1]
        super(CategoricalVariable, self).__init__(name, domain, bounds, 'categorical', domain.shape[0])
        
    def get_one_hot(self, x):
        '''
        Return one hot encoding for the given value based on the position
        in bounds
        '''
        if x in self.domain:
            one_hot = np.zeros((x.shape[0],self.domain.shape[0]))
            one_hot[self.domain == x] = 1
        
        return one_hot
    
    def get_value(self, x):
        '''
        Return the value associated to the one hot encoding given
        '''
        return np.array([self.domain[arr == 1] for arr in x])
    
    def get_rounded(self, x, norm=False):
        '''
        Return the one hot encoding with 1 on the max value 
        '''
        max_idx = np.argmax(x, axis=1)
        fix = np.zeros_like(x)
        for i, row in zip(max_idx, fix):
            row[i] = 1.0
        return fix