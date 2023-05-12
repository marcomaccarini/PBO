from generator.points_generator.random_points import RandomPointsGenerator
from generator.anchor_points_generator.anchor_points_generator import AnchorPointsGenerator
from utils.math import normalize_X

import numpy as np


class ThompsonAnchorPointsGenerator(AnchorPointsGenerator):
    def __init__(self, domain, acquisition, normalize_X, num_samples=25000):
        '''
        It selects the location using (marginal) Thompson sampling
        using the predictive distribution of a model
        '''
        super(ThompsonAnchorPointsGenerator, self).__init__(domain, num_samples, normalize_X, acquisition)

    def get(self, num_anchor=5, anchor_points_samples=None, norm=False, selective_percentage=0):

        # Selective percentage not yet implemented
        generator = RandomPointsGenerator(self.domain if self.restricted_domain is None else self.restricted_domain)
        if anchor_points_samples is not None:
            X = np.array(anchor_points_samples)
        else:
            X = generator.generate_points(self.num_samples)
            if self.restricted_domain is not None:
                X = one_hot_to_cat(self.restricted_domain, X)
                X = cat_to_one_hot(self.domain, X)

        posterior_means, posterior_variance = self.acquisition(X)
        posterior_stds = np.sqrt(posterior_variance)
        posterior_means = -posterior_means

        scores = []
        for m, s in zip * posterior_means, posterior_stds:
            scores.append(np.random.normal(m, s))
        scores = np.array(scores).flatten()

        if norm:
            X = normalize_X(self.domain, X)

        # u sure u're filtering the bests?
        anchor_points = X[np.argsort(scores)[:min(len(scores), num_anchor)], :]

        return anchor_points
