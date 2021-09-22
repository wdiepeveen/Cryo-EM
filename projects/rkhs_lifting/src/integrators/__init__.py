import logging

import numpy as np

from projects.rkhs_lifting.src.integrators.base import SO3_Integrator
from projects.rkhs_lifting.src.kernels import RKHS_Kernel

logger = logging.getLogger(__name__)


class RKHS_Density_Integrator:
    """Integration against a density build up from RKHS kernels"""
    def __init__(self, base_integrator, kernel, dtype=np.float32):

        self.dtype = dtype

        assert type(base_integrator) == SO3_Integrator
        self.base_integrator = base_integrator
        assert type(kernel) == RKHS_Kernel
        self.kernel = kernel

        self.n = base_integrator.n
        self.manifold = base_integrator.manifold
        self.weight_matrix = base_integrator.weight_matrix

    @property
    def angles(self):
        return self.base_integrator.angles

    @property
    def rots(self):
        return self.base_integrator.rots

    @property
    def quaternions(self):
        return self.base_integrator.quaternions

    def coeffs_to_density(self, coeffs, cap_density=True):
        density = self.kernel.matrix_mult(coeffs)
        if cap_density:
            density = np.maximum(0, density)
        # TODO also normalize with l1-norm?

        return density.astype(self.dtype)

    def coeffs_to_weights(self, coeffs, cap_density=True):
        density = self.coeffs_to_density(coeffs, cap_density=cap_density)
        weights = self.base_integrator.weigh_integrands(density)
        return weights.astype(self.dtype)

    # def proj(self, coeffs):  # TODO: probably in other class
    #     weights = self.coeffs2weights(coeffs)
    #     np.clip(weights, 0.0, 1.0, out=weights)
    #     weights /= weights.sum(axis=1)[:, None]
    #     return self.manifold.mean(self.quaternions[None, None], weights[None])[0, 0]
    #
    # def MAP_project(self, coeffs):  # TODO
    #     weights = self.coeffs2weights(coeffs)
    #     np.clip(weights, 0.0, 1.0, out=weights)
    #     weights /= weights.sum(axis=1)[:, None]
