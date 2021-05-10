import numpy as np

from scipy.spatial.transform import Rotation as R
import spherical

from solvers.lifting.integration import Integrator


class IcosahedronIntegrator(Integrator):

    def __init__(self,
                 ell_max=3,
                 dtype=np.float32,
                 ):

        super().__init__(dtype=dtype, n=60, ell=int((2 * ell_max + 1) * (2 * ell_max + 2) * (2 * ell_max + 3) / 6), t=5)

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

        # Copmpute U for all the rotations
        U = np.zeros((self.n, self.ell), dtype=complex)
        wigner = spherical.Wigner(ell_max)
        quats = self.quaternions
        for i in range(self.n):
            g = quats[i]
            U[i] = wigner.D(g)

        # # Check that U integrates correctly
        # integrals = np.real(np.sum(U, axis = 0))/60
        # print("integrals U = {}".format(integrals))

        # Compute V
        V = np.zeros((self.ell, self.ell), dtype=complex)
        for l in range(ell_max + 1):
            for m in range(-l, l + 1):
                for n in range(-l, l + 1):
                    if m==0 and n==0:
                        index = wigner.Dindex(l, m, n)
                        V[index,index] = 1
                    elif m>0 or (m==0 and n>0):
                        index = wigner.Dindex(l, m, n)
                        V[index, index] = 1
                        indexx = wigner.Dindex(l, -m, -n)
                        V[index, indexx] = (-1)**(m-n) * 1j
                    elif m<0 or (m==0 and n<0):
                        index = wigner.Dindex(l, m, n)
                        V[index, index] = -1j
                        indexx = wigner.Dindex(l, -m, -n)
                        V[index, indexx] = (-1) ** (m - n)

        UV = U@V
        UV /= 60
        # print(np.imag(UV))
        self.b2w = np.real(UV).T.astype(self.dtype)

    def coeffs2weights(self, coeffs, cap_weights=True):
        weights = coeffs@self.b2w
        if cap_weights:
            weights = np.maximum(0, weights)

        return weights.astype(self.dtype)

    def weights2coeffs(self, weights):
        coeffs = weights@self.b2w.T

        return coeffs.astype(self.dtype)
