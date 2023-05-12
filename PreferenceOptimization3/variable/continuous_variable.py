#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 27 15:03:27 2018

@author: Stefano Toniolo
"""

from .variable import Variable

class ContinuousVariable(Variable):
    
    def __init__(self, name, domain):
        super(ContinuousVariable, self).__init__(name, domain, domain, 'continuous', 1)
        
    def get_rounded(self, x, norm=False):
        return x