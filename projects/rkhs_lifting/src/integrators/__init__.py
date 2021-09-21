import logging

import numpy as np
import quaternionic

from scipy.spatial.transform import Rotation as R

from projects.rkhs_lifting.src.manifolds.so3 import SO3
from projects.rkhs_lifting.src.integrators.so3_haar import SO3_Integrator
from projects.rkhs_lifting.src.kernels import RKHS_Kernel

logger = logging.getLogger(__name__)


class RKHS_Density_Integrator:
    """Integration against a density build up from RKHS kernels"""
    def __init__(self, integrator, kernel, dtype=np.float32):

        assert type(integrator) == SO3_Integrator
        self.integrator = integrator
        assert type(kernel) == RKHS_Kernel
        self.kernel = kernel


        # self.n = quaternions.shape[0]
        # print("n = {}".format(self.n))
        # self._points = None
        # self.quaternions = quaternions  # TODO check whether this is alright (we use the setter before we define it)
        #
        # self.kernel = kernel
        #
        # self.manifold = SO3()  #quats=self.quaternions)  # TODO check whether this is alright. We shouldn't need it
        #
        # logger.info("Construct distance matrix")
        # print("Construct distance matrix")
        # distances = self.manifold.dist(self.quaternions[None, :, :], self.quaternions[None, :, :])[0]
        # W = self.kernel(distances)
        # Wt = (np.abs(W) >= threshold) * W  # TODO should we make this sparse?
        # self.b2w = Wt / self.n


    # @property
    # def angles(self):
    #     return self._points.as_euler("ZYZ").astype(self.dtype)
    #
    # @angles.setter
    # def angles(self, values):
    #     self._points = R.from_euler("ZYZ", values)
    #
    # @property
    # def rots(self):
    #     return self._points.as_matrix().astype(self.dtype)
    #
    # @rots.setter
    # def rots(self, values):
    #     self._points = R.from_matrix(values)
    #
    # @property
    # def quaternions(self):
    #     quats = np.roll(self._points.as_quat().astype(self.dtype),1,axis=-1)
    #     sign_s = np.sign(quats[:, 0])
    #     sign_s[sign_s == 0] = 1
    #     return quaternionic.array(sign_s[:, None] * quats).normalized.ndarray
    #
    # @quaternions.setter
    # def quaternions(self, values):
    #     quats = quaternionic.array(np.roll(values,-1,axis=-1)).normalized.ndarray
    #     self._points = R.from_quat(quats)

    def coeffs2weights(self, coeffs, cap_weights=True):
        weights = coeffs @ self.b2w
        if cap_weights:
            weights = np.maximum(0, weights)

        return weights.astype(self.dtype)

    def weights2coeffs(self, weights):
        coeffs = weights @ self.b2w  # (.T) => since symmetric

        return coeffs.astype(self.dtype)

    # def proj(self, coeffs):  # TODO
    #     weights = self.coeffs2weights(coeffs)
    #     np.clip(weights, 0.0, 1.0, out=weights)
    #     weights /= weights.sum(axis=1)[:, None]
    #     return self.manifold.mean(self.quaternions[None, None], weights[None])[0, 0]
    #
    # def MAP_project(self, coeffs):  # TODO
    #     weights = self.coeffs2weights(coeffs)
    #     np.clip(weights, 0.0, 1.0, out=weights)
    #     weights /= weights.sum(axis=1)[:, None]
