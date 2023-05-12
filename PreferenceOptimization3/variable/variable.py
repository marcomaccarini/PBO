#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 13 10:06:01 2018

@author: Stefano Toniolo
"""

class Variable(object):
    
    def __init__(self, name, domain, bounds, var_type, dim):
        self.name = name
        self.domain = domain
        self.bounds = bounds
        self.var_type = var_type
        self.dim = dim
    
    def get_name(self):
        return self.name
    
    def get_type(self):
        return self.var_type
    
    def get_bounds(self):
        return self.bounds
    
    def get_domain(self):
        return self.domain
    
    def get_dim(self):
        return self.dim
    
    def get_normalized(self):
        raise NotImplementedError("Function not implemented")
    
    def get_rounded(self):
        raise NotImplementedError("Function not implemented")