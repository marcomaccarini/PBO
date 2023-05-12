#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb  17 10:20:05 2022

@author: Marco Maccarini
"""

import time
import warnings

from abc import ABC, abstractmethod


class RadialBasisFunction(ABC):
    """
    Abstract class representing the basic structure each rbf class has to implement
    """

    @abstractmethod
    def rbf(self, eps, d):
        """
        Methods that implement the optimization procedure
        :param eps: epslon
        :param d: distance
        :return:
        """
        raise NotImplementedError("Under development")
