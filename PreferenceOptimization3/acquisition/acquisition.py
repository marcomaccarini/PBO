#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov  2 18:16:30 2018

@author: Stefano Toniolo (IDSIA)
"""

import numpy as np
import sys
from abc import ABC, abstractmethod
from PreferenceOptimization3.utils.math import denormalize_X, one_hot_to_cat


class Acquisition(ABC):
    """
    Abstract Acquisition class containing concrete method to manage the
    acquisition function ans 2 abstract method to compute the function and
    its gradient if provided.
    """

    def __init__(self, model, grad):
        """
        Class initializer
        :param grad: Gradient
        """
        self.grad = grad
        self.model = model
        self.fvars_x = None
        self.transfer_learning_model = None
        self.constraints = None
        self.exploration = True

        self.dd = None

    def get_acquisition(self, x):
        """
        Compute the acquisition function in x.
        :param x: Where the acquisition function has to be evaluated
        :return: Acquisition function in x (negative because it is a maximization problem)
        """
        x = x.reshape(-1, self.model.X.shape[1])
        # if self.model.normalize_X:
        #    x_unnorm = denormalize_X(self.model.fvars_x, x)
        # else:
        #    x_unnorm = x

        x_acq = self._compute_acquisition(x)  # .flatten()
        # x_acq = self._compute_acquisition(x_unnorm).flatten()
        # x_mask = self._mask_constraints(x_unnorm).flatten()
        # x_acq[x_mask == 0] = - (sys.maxsize - 1)

        return x_acq

    def get_acquisition_grad(self, x):
        """
        Compute the gradient of the acquisition function.
        :param x: Where the gradient has to be evaluated
        :return: Gradient of the acquisition function evaluated in x (negative because it is a maximization problem)
        """
        x = x.reshape(-1, self.model.X.shape[1])
        if self.model.normalize_X:
            x_unnorm = denormalize_X(self.model.fvars_x, x)
        else:
            x_unnorm = x
        x_acq_grad = self._compute_acquisition_grad(x_unnorm)
        return -x_acq_grad * self._mask_constraints(x_unnorm)

    def grad_support(self):
        """
        Check if the gradient is supported for the current optimization technique.
        :return: Boolean
        """
        return self.grad

    def set_constraints(self, constraints):
        """
        Set optimization constraints
        :param constraints: Constranits to be set.
        :return:
        """
        self.constraints = constraints

    def set_model(self, model):
        """
        Set the gaussian model used to compute the acquisition function.
        :param model: Gaussian model
        :return:
        """
        self.model = model

    def set_exploration(self, exploration=True):
        """
        Set the mode to compute the acquisition function (exploration or exploitation).
        :param exploration: exploration flag
        :return:
        """
        self.exploration = exploration

    def _mask_constraints(self, x):
        """
        Create a mask based on the constraints used to limit the acquisition function's space.
        Only inequality constraints supported at the moment.
        The mask set the acquisition function to 0 where the acquisition points are outside the feasible space.
        :param x: Acquisition points
        :return: Mask
        """
        x = np.atleast_2d(x)
        x_mask = np.ones((x.shape[0], 1))
        if self.constraints is not None:
            for c in self.constraints:
                yvals = []
                for point in x:
                    yvals.append(c(one_hot_to_cat(self.model.fvars_x, point.reshape(1, -1)).flatten(), self.model))
                mask = (np.array(yvals) > 0) * 1
                x_mask *= mask.reshape((x.shape[0], 1))
        return x_mask

    @abstractmethod
    def _compute_acquisition(self, x):
        """
        Abstract method overridden by each acquisition function.
        This method contains the mathematical implementation of the acquisition function.
        :param x: Acquisition points
        :return: Acquisition function evaluations
        """
        raise NotImplementedError("Acquisition function non supported")

    @abstractmethod
    def _compute_acquisition_grad(self, x):
        """
        Abstract method overridden by each acquisition function.
        This method contains the mathematical implementation of the acquisition function's gradient.
        :param x: Acquisition points
        :return: Gradient evaluations
        """
        raise NotImplementedError("Acquisition function not supported")
