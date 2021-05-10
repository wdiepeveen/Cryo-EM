
import numpy as np

import spherical

from solvers.lifting.integration import Integrator


class UniformIntegrator(Integrator):

    def __init__(self,
                 ell_max=3,
                 n=1000,
                 dtype=np.float32,
                 ):

        super().__init__(dtype=dtype, n=n, ell=int((2 * ell_max + 1) * (2 * ell_max + 2) * (2 * ell_max + 3) / 6), t=np.inf)

        # Compute Euler angles
        angles = np.zeros((n, 3))

        angles[:, 0] = np.random.random(n) * 2 * np.pi
        angles[:, 1] = np.arccos(2 * np.random.random(n) - 1)
        angles[:, 2] = np.random.random(n) * 2 * np.pi

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
        # integrals = np.real(np.sum(U, axis=0))/n
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
        UV /= n
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
