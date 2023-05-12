from PreferenceOptimization3.generator.points_generator.random_points import RandomPointsGenerator
from PreferenceOptimization3.generator.anchor_points_generator.anchor_points_generator import AnchorPointsGenerator
from PreferenceOptimization3.utils.math import normalize_X, denormalize_X  # , cat_to_one_hot, one_hot_to_cat

import numpy as np


class RandomAnchorPointsGenerator(AnchorPointsGenerator):
    def __init__(self, domain, acquisition, normalize_X, num_samples=1000):
        super(RandomAnchorPointsGenerator, self).__init__(domain, num_samples, normalize_X, acquisition)

    def get(self, num_anchor=5, anchor_points_samples=None, norm=False, selective_percentage=0):

        generator = RandomPointsGenerator(self.domain if self.restricted_domain is None else self.restricted_domain)
        X = generator.generate_points(self.num_samples)
        if norm:
            X = normalize_X(self.domain, X)
        # todo: normalization?

        # scores = self.acquisition(X)
        scores = self.acquisition(X)

        if selective_percentage == 0:
            anchor_points = X[np.argsort(scores)[:min(len(scores), num_anchor)], :]
            return anchor_points
        else:
            if norm:
                normalized_X = X
            else:
                normalized_X = normalize_X(self.domain, X)

            anchor_points = []
            sorted_points = normalized_X[np.argsort(scores), :]
            anchor_points.append(sorted_points[0])

            for point in sorted_points:
                select = True
                for anchor in anchor_points:
                    for dim in range(len(self.domain)):
                        # if the difference on the domain is under the given percentage
                        if abs(anchor[dim] - point[dim]) < selective_percentage:
                            select = False
                            break
                # all dimensions should respect the d
                if select:
                    anchor_points.append(point)
                if len(anchor_points) >= num_anchor:
                    break

            if not norm:
                anchor_points = denormalize_X(self.domain, np.array(anchor_points))

            return anchor_points
