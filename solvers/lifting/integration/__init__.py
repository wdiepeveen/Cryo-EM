import logging

import numpy as np
import spherical
from scipy.spatial.transform import Rotation as R
from scipy.sparse import csc_matrix

from solvers.lifting.integration.discretized_manifolds.so3 import SO3

logger = logging.getLogger(__name__)


class Integrator:
    def __init__(self, dtype=np.float32, n=None, ell_max=None, t=None):
        self.dtype = dtype
        assert n is not None
        self.n = n
        self.ell_max = ell_max
        if not t == np.inf:
            assert ell_max <= t

        self.ell = int((2 * ell_max + 1) * (2 * ell_max + 2) * (2 * ell_max + 3) / 6)
        self.t = t

        self._points = None
        self.manifold = None
        self.U = None
        self.V = None
        self.b2w = None

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
        quats = np.roll(self._points.as_quat().astype(self.dtype),1,axis=-1)
        sign_s = np.sign(quats[:, 0])
        sign_s[sign_s == 0] = 1
        return sign_s[:, None] * quats

    @quaternions.setter
    def quaternions(self, values):
        quats = np.roll(values,-1,axis=-1)
        self._points = R.from_quat(quats)

    def coeffs2weights(self, coeffs, cap_weights=True):
        weights = coeffs @ self.b2w
        if cap_weights:
            weights = np.maximum(0, weights)

        return weights.astype(self.dtype)

    def weights2coeffs(self, weights):
        coeffs = weights @ self.b2w.T

        return coeffs.astype(self.dtype)

    def proj(self, coeffs):
        weights = self.coeffs2weights(coeffs)
        np.clip(weights, 0.0, 1.0, out=weights)
        weights /= weights.sum(axis=1)[:, None]
        return self.manifold.mean(self.quaternions[None, None], weights[None])[0, 0]

    def MAP_project(self, coeffs):
        weights = self.coeffs2weights(coeffs)
        np.clip(weights, 0.0, 1.0, out=weights)
        weights /= weights.sum(axis=1)[:, None]

        # TODO get argmax quaternion for all (can use what we do in mean)

        # TODO make sure that we are not on a degenerate point

        # TODO compute J1

        # TODO compute J2

        # TODO einsum

        # TODO project onto tangent space (see how we want to do this for degen ~ parallel transport?)\

        # TODO exponential step after choosing step size (use 1/ell_max)

    def initialize_manifold(self):
        self.manifold = SO3(quats=self.quaternions)

    def initialize_b2w(self):
        # Copmpute U for all the rotations
        logger.info("Construct U matrix")
        # U = np.zeros((self.n, self.ell), dtype=np.complex64)
        wigner = spherical.Wigner(self.ell_max)
        quats = self.quaternions
        U = wigner.D(quats).astype(np.complex64)
        # for i in range(self.n):
        #     if self.n > 5000 and i % 5000 == 0:
        #         logger.info("Computing U matrix | progress: {} %".format(np.round(i / self.n * 100)))
        #     g = quats[i]
        #     U[i] = wigner.D(g)  # TODO this should not be necessary -> we should be able to input array

        # # Check that U integrates correctly
        integrals = np.real(np.sum(U, axis=0)) / self.n
        logger.info("integrals U = {}".format(integrals))
        logger.info("l_inf error = {}".format(np.max(np.abs(integrals - np.eye(1, self.ell)[0]))))

        self.U = csc_matrix(U)

        # Compute V
        logger.info("Construct V matrix")
        Vrow = []
        Vcol = []
        Vdata = []
        for l in range(self.ell_max + 1):
            for m in range(-l, l + 1):
                for n in range(-l, l + 1):
                    if m == 0 and n == 0:
                        index = wigner.Dindex(l, m, n)
                        # V[index,index] = 1
                        Vrow.append(index)
                        Vcol.append(index)
                        Vdata.append(1)
                    elif m > 0 or (m == 0 and n > 0):
                        index = wigner.Dindex(l, m, n)
                        # V[index, index] = 1
                        Vrow.append(index)
                        Vcol.append(index)
                        Vdata.append(1)

                        indexx = wigner.Dindex(l, -m, -n)
                        # V[index, indexx] = (-1)**(m-n) * 1j
                        Vrow.append(index)
                        Vcol.append(indexx)
                        Vdata.append((-1) ** (m - n) * 1j)
                    elif m < 0 or (m == 0 and n < 0):
                        index = wigner.Dindex(l, m, n)
                        # V[index, index] = -1j
                        Vrow.append(index)
                        Vcol.append(index)
                        Vdata.append(-1j)

                        indexx = wigner.Dindex(l, -m, -n)
                        # V[index, indexx] = (-1) ** (m - n)
                        Vrow.append(index)
                        Vcol.append(indexx)
                        Vdata.append((-1) ** (m - n))

        self.V = csc_matrix((Vdata, (Vrow, Vcol)), shape=(self.ell, self.ell), dtype=np.complex64)

        logger.info("Construct B2W matrix")
        UV = self.U @ self.V
        UV /= self.n

        self.b2w = np.real(UV).T.astype(self.dtype)
