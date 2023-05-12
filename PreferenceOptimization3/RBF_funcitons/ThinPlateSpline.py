from .radial_basis_function import RadialBasisFunction
import numpy as np


class ThinPlateSpline(RadialBasisFunction):

    def rbf(self, eps, d):
        app = eps*d
        return app**2 * np.log(app)
