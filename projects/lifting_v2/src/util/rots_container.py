import numpy as np
import quaternionic

from scipy.spatial.transform import Rotation as R

class RotsContainer:

    def __init__(self, n, dtype=np.float32):
        self.dtype = dtype
        self.n = n

        self._points = None

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
