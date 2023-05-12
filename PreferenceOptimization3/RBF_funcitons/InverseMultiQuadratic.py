from .radial_basis_function import RadialBasisFunction
import numpy as np


class InverseMultiQuadratic(RadialBasisFunction):

    def rbf(self, eps, d):
        app = (eps*d)**2
        return 1/(np.sqrt(1+app))
