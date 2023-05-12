# -*- coding: utf-8 -*-

from abc import ABC, abstractmethod


class AnchorPointsGenerator(ABC):
    def __init__(self, domain, num_samples, normalize_X, acquisition):
        self.domain = domain
        self.num_samples = num_samples
        self.normalize_X = normalize_X
        self.acquisition = acquisition
        self.restricted_domain = None

    @abstractmethod
    def get(self, num_anchor, anchor_points_samples=None, norm=False, selective_percentage=0):
        raise NotImplementedError("Method not implemented")

    def set_restricted_domain(self, restricted_domain):
        self.restricted_domain = restricted_domain