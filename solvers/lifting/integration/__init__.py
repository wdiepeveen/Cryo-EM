import numpy as np
from scipy.spatial.transform import Rotation as R


class Integrator:
    def __init__(self, dtype=np.float32, n=None, ell=None, t=None):
        self.dtype = dtype
        assert n is not None
        self.n = n
        if not t == np.inf:
            assert ell <= int((2 * t + 1) * (2 * t + 2) * (2 * t + 3) / 6)

        self.ell = ell
        self.t = t

        self._points = None  # must be rotation matrices
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
        return self._points.as_quat().astype(self.dtype)

    @quaternions.setter
    def quaternions(self, values):
        self._points = R.from_quat(values)

    def coeffs2weights(self, coeffs):
        raise NotImplementedError(
            "Subclasses should implement this and return an 2D Array object"
        )

    def weights2coeffs(self, weights):
        raise NotImplementedError(
            "Subclasses should implement this and return an 2D Array object"
        )