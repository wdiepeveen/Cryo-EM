
import numpy as np

import spherical

from solvers.lifting.integration import Integrator


class UniformIntegrator(Integrator):

    def __init__(self,
                 ell_max=3,
                 n=1000,
                 dtype=np.float32,
                 ):

        super().__init__(dtype=dtype, n=n, ell_max=ell_max, t=np.inf)

        # TODO sample random quaternions instead
        # Compute Euler angles
        angles = np.zeros((n, 3))

        angles[:, 0] = np.random.random(n) * 2 * np.pi
        angles[:, 1] = np.arccos(2 * np.random.random(n) - 1)
        angles[:, 2] = np.random.random(n) * 2 * np.pi

        # Compute points
        self.angles = angles
        self.initialize_manifold()
        self.initialize_b2w()
