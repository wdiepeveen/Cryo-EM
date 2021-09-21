import numpy as np

from projects.rkhs_lifting.src.integrators.so3_haar import SO3_Integrator


class SD60(SO3_Integrator):
    """Icosahedron integrator"""

    def __init__(self, dtype=np.float32):

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

        super().__init__(angles, representation="angles", dtype=dtype)
