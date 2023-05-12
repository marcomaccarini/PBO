from .radial_basis_function import RadialBasisFunction
import numpy as np


class Gaussian(RadialBasisFunction):

    def rbf(self, eps, d):
        app = (eps*d)**2
        return np.exp(-app)
