from .radial_basis_function import RadialBasisFunction
import numpy as np


class Linear(RadialBasisFunction):

    def rbf(self, eps, d):
        app = eps*d
        return app
