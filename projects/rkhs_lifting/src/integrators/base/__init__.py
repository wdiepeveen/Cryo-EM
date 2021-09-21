import logging

import numpy as np
import quaternionic

from scipy.spatial.transform import Rotation as R

from projects.rkhs_lifting.src.manifolds.so3 import SO3

logger = logging.getLogger(__name__)


class SO3_Integrator:
    """Integrators of the Haar measure"""

    def __init__(self, verts, representation="quaternions", weights=None, dtype=np.float32):

        self.dtype = dtype
        self.n = verts.shape[0]

        self.manifold = SO3()
        self._points = None

        if representation == "quaternions":
            self.quaternions = verts
        elif representation == "angles":
            self.angles = verts
        else:
            raise NotImplementedError(
                "representation should be either 'quaternions' or 'angles' format"
            )

        if weights is not None:
            k=0
        else:
            k=1  # use 1/self.n otherwise

        # TODO make some object that handles the weights

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

    def coeffs2weights(self, coeffs, cap_weights=True):  # TODO fix
        weights = coeffs @ self.b2w
        if cap_weights:
            weights = np.maximum(0, weights)

        return weights.astype(self.dtype)

    def weights2coeffs(self, weights):  # TODO fix
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
