from .radial_basis_function import RadialBasisFunction
import numpy as np


class Multiquadratic(RadialBasisFunction):

    def rbf(self, eps, d):
        app = (eps*d)**2
        return np.sqrt(1+app)
