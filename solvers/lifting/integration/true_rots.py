import numpy as np

from scipy.spatial.transform import Rotation as R
import spherical

from solvers.lifting.integration import Integrator


class TrueRotsIntegrator(Integrator):

    def __init__(self,
                 rots=None,
                 dtype=np.float32,
                 ):

        super().__init__(dtype=dtype, n=rots.shape[0], ell_max=0, t=np.inf)
        self.ell = rots.shape[0]

        # Compute Euler angles

        self.rots = rots
        self.initialize_manifold()
        self.b2w = np.eye(self.ell, self.n, dtype=self.dtype)

    def coeffs2weights(self, coeffs, cap_weights=True):

        return np.eye(self.ell, self.n, dtype=self.dtype)

    def weights2coeffs(self, weights):

        return np.eye(self.ell, self.n, dtype=self.dtype)
