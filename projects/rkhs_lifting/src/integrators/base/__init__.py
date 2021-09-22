import logging

import numpy as np
import quaternionic

from scipy.sparse import diags
from scipy.spatial.transform import Rotation as R

from projects.rkhs_lifting.src.manifolds.so3 import SO3

logger = logging.getLogger(__name__)


class SO3_Integrator:
    """Integration with respect to the Haar measure"""

    def __init__(self, quaternions, weights=None, dtype=np.float32):

        self.dtype = dtype
        self.n = quaternions.shape[0]

        self.manifold = SO3()

        self._points = None
        self.quaternions = quaternions

        if weights is not None:
            assert self.n == weights.shape[0]
            self.weight_matrix = diags(weights)
        else:
            self.weight_matrix = diags(1/self.n * np.ones(self.n,))


    @property
    def angles(self):
        return self._points.as_euler("ZYZ").astype(self.dtype)

    @angles.setter
    def angles(self, values):
        self._points = R.from_euler("ZYZ", values)

    @property
    def rots(self):
        return self._points.as_matrix().astype(self.dtype)

    @rots.setter
    def rots(self, values):
        self._points = R.from_matrix(values)

    @property
    def quaternions(self):
        quats = np.roll(self._points.as_quat().astype(self.dtype), 1, axis=-1)
        sign_s = np.sign(quats[:, 0])
        sign_s[sign_s == 0] = 1
        return quaternionic.array(sign_s[:, None] * quats).normalized.ndarray

    @quaternions.setter
    def quaternions(self, values):
        quats = quaternionic.array(np.roll(values, -1, axis=-1)).normalized.ndarray
        self._points = R.from_quat(quats)

    def weigh_integrands(self, integrands):
        weighted_integrands = self.weight_matrix @ integrands
        return weighted_integrands.astype(self.dtype)

    def integrate(self, integrands):
        return np.sum(self.weigh_integrands(integrands), axis=0)

