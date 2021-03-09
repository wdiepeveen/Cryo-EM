import numpy as np
from scipy.spatial.transform import Rotation as R


class Integrator:
    def __init__(self, dtype=np.float32):
        self.dtype = dtype
        self._points = None  # must be rotation matrices

    @property
    def angles(self):
        return self._points.as_euler().astype(self.dtype)

    @angles.setter
    def points(self, values):
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
        self._points = R.from_quat(values).astype(self.dtype)

    def integrate(self, values):
        raise NotImplementedError(
            "Subclasses should implement this"
        )
