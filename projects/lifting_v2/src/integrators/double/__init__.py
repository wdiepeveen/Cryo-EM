import logging

import numpy as np
import quaternionic

from scipy.spatial.transform import Rotation as R

from projects.lifting_v2.src.manifolds.so3 import SO3

logger = logging.getLogger(__name__)


class Double_SO3_Integrator:
    """
    Discretized Integration of the form \sum_\rot \int_SO3 F(\rot') d\measure_\rot(\rot') \beta^\rot, e.g.,:
    * \measure_\rot = \dirac_\rot(\cdot) => \sum_\rot F(\rot) \beta^\rot
    ** Default, no additional inputs required
    * \measure_\rot = K(\rot,\cdot)d\rot' => \sum_\rot \int_SO3 F(\rot') K(\rot,\rot') d\rot' \beta^\rot
    -> \sum_\rot \sum_\rot' w_\rot' F(\rot') K(\rot,\rot') \beta^\rot
    ** Provide kernel and if necessary weights and inner_quaternions
    * \measure_\rot = \sum_c 1/C \delta_c\rot => 1\C \sum_c \sum_\rot F(c\rot) \beta^\rot
    ** Provide kernel, weights and inner_quaternions
    """

    def __init__(self, outer_quaternions, kernel=None, weights=None, inner_quaternions=None, dtype=np.float32):

        self.dtype = dtype
        self.kernel = kernel  # K or symmetrisation C

        self.n = outer_quaternions.shape[0]
        if inner_quaternions is not None:
            assert kernel is not None  # TODO Can have cases where we do have inner rots, but no kernel
            self.nn = inner_quaternions.shape[0]
        else:
            self.nn = self.n

        self.manifold = SO3()

        self._points = None  # \rot
        self.quaternions = outer_quaternions

        self._pointspoints = None  # \rot'
        if inner_quaternions is not None:
            assert kernel is not None
            self.quaternionsquaternions = inner_quaternions
        else:
            self.quaternionsquaternions = self.quaternions

        if weights is not None:
            assert kernel is not None
            assert self.nn == weights.shape[0] and len(weights.shape) == 1
            self.weights = weights[:, None]
        else:
            self.weights = (1 / self.nn * np.ones(self.nn, ))[:, None]

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
        return quaternionic.array(sign_s[:, None] * quats).normalized.ndarray.astype(self.dtype)

    @quaternions.setter
    def quaternions(self, values):
        quats = quaternionic.array(np.roll(values, -1, axis=-1)).normalized.ndarray
        self._points = R.from_quat(quats)

    # Inner integration

    @property
    def anglesangles(self):
        return self._pointspoints.as_euler("ZYZ").astype(self.dtype)

    @anglesangles.setter
    def anglesangles(self, values):
        self._pointspoints = R.from_euler("ZYZ", values)

    @property
    def rotsrots(self):
        return self._pointspoints.as_matrix().astype(self.dtype)

    @rotsrots.setter
    def rotsrots(self, values):
        self._pointspoints = R.from_matrix(values)

    @property
    def quaternionsquaternions(self):
        quats = np.roll(self._points.as_quat().astype(self.dtype), 1, axis=-1)
        sign_s = np.sign(quats[:, 0])
        sign_s[sign_s == 0] = 1
        return quaternionic.array(sign_s[:, None] * quats).normalized.ndarray.astype(self.dtype)

    @quaternionsquaternions.setter
    def quaternionsquaternions(self, values):
        quats = quaternionic.array(np.roll(values, -1, axis=-1)).normalized.ndarray
        self._pointspoints = R.from_quat(quats)

    # def weigh_integrands(self, integrands):
    #     if len(integrands.shape) == 1:
    #         weights = self.weights.squeeze()
    #     else:
    #         weights = self.weights
    #     weighted_integrands = weights * integrands
    #     return weighted_integrands.astype(self.dtype)
    #
    # def integrate(self, integrands):
    #     return np.sum(self.weigh_integrands(integrands), axis=0)

    def coeffs_to_integrand_weights(self, coeffs):
        if self.kernel is not None:
            pre_weights = self.kernel.matrix_mult(coeffs)
            return (self.weights * pre_weights).astype(self.dtype)
        else:
            return coeffs

    def integrands_to_coeff_weights(self, integrands):
        if self.kernel is not None:
            pre_weights = (self.weights * integrands).astype(self.dtype)
            return self.kernel.adjoint_matrix_mult(pre_weights)
        else:
            return integrands
