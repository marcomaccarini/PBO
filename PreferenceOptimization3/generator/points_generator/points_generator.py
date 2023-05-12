#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec  2 16:57:46 2018

@author: Stefano Toniolo
"""

from abc import ABC, abstractmethod


class PointsGenerator(object):

    @abstractmethod
    def generate_points(self, num_points):
        raise NotImplementedError("Under developement")
