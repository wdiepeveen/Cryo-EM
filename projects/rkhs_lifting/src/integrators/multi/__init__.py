import logging

import numpy as np
import quaternionic

# from scipy.sparse import diags
from scipy.spatial.transform import Rotation as R

from projects.rkhs_lifting.src.manifolds.so3 import SO3

logger = logging.getLogger(__name__)


class SO3_Multi_Integrator:
    """Integration with respect to the Haar measure with channel wise different integration schemes"""

    def __init__(self, quaternions=None, sep_dists=None, mesh_norms=None, tri_dists=None, weights=None,
                 dtype=np.float32):

        self.dtype = dtype

        assert len(quaternions.shape) == 3
        self.n = quaternions.shape[0]
        self.N = quaternions.shape[1]

        self.sep_dists = sep_dists  # Separation distance
        self.mesh_norms = mesh_norms  # Mesh norm
        self.tri_dists = tri_dists  # Maximum distance in the triangulation

        self.manifold = SO3()

        self._points = None
        self.quaternions = quaternions

        if weights is not None:
            assert self.n == weights.shape[0] and len(weights.shape) == 1
            self.weights = weights[:, None]
            # self.weight_matrix = diags(weights)
        else:
            # self.weight_matrix = diags(1 / self.n * np.ones(self.n, ))
            self.weights = (1 / self.n * np.ones(self.n, ))[:, None]

    @property
    def angles(self):
        return self._points.as_euler("ZYZ").astype(self.dtype).reshape((self.n,self.N,3))

    @angles.setter
    def angles(self, values):
        self._points = R.from_euler("ZYZ", values.reshape((self.n * self.N, 3)))

    @property
    def rots(self):
        return self._points.as_matrix().astype(self.dtype).reshape((self.n,self.N,3,3))

    @rots.setter
    def rots(self, values):
        self._points = R.from_matrix(values.reshape((self.n * self.N, 3, 3)))

    @property
    def quaternions(self):
        quats = np.roll(self._points.as_quat().astype(self.dtype), 1, axis=-1)
        sign_s = np.sign(quats[:, 0])
        sign_s[sign_s == 0] = 1
        return (quaternionic.array(sign_s[:, None] * quats).normalized.ndarray.astype(self.dtype)).reshape((self.n,self.N,4))

    @quaternions.setter
    def quaternions(self, values):
        quats = quaternionic.array(np.roll(values, -1, axis=-1)).normalized.ndarray
        self._points = R.from_quat(quats.reshape((self.n * self.N, 4)))

    # TODO check whether this still makes sense
    def weigh_integrands(self, integrands):
        weighted_integrands = self.weights * integrands
        # weighted_integrands = self.weight_matrix @ integrands
        return weighted_integrands.astype(self.dtype)

    def integrate(self, integrands):
        return np.sum(self.weigh_integrands(integrands), axis=0)
