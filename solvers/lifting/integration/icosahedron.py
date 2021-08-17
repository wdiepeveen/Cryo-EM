import numpy as np

from scipy.spatial.transform import Rotation as R
import spherical

from solvers.lifting.integration import Integrator


class IcosahedronIntegrator(Integrator):

    def __init__(self,
                 ell_max=3,
                 dtype=np.float32,
                 ):

        super().__init__(dtype=dtype, n=60, ell_max=ell_max, t=5)

        # Compute Euler angles
        angles = np.zeros((60, 3))

        for i in range(12):
            if i == 0:
                phi = 0
                theta = 0
            elif i <= 5:
                phi = 2 / 5 * i * np.pi
                theta = np.arctan(2)
            elif i <= 10:
                phi = 2 / 5 * (i - 5) * np.pi + 1 / 5 * np.pi
                theta = np.pi - np.arctan(2)
            else:
                phi = 0
                theta = np.pi

            for j in range(5):
                if i == 0 or i == 11:
                    c = 0
                else:
                    c = 1 / 5 * np.pi

                omega = 2 / 5 * j * np.pi + c
                if omega - phi < 0:
                    psi = 2 * np.pi + omega - phi
                else:
                    psi = omega - phi

                angles[i * 5 + j] = np.array([phi, theta, psi])

        # Compute points
        self.angles = angles
        self.initialize_manifold()
        self.initialize_b2w()
