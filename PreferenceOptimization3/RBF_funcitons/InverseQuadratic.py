from .radial_basis_function import RadialBasisFunction
import numpy as np


class InverseQuadratic(RadialBasisFunction):

    def rbf(self, eps, d):
        app = (eps*d)**2
        return 1/(1+app)
