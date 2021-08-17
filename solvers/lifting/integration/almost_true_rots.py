import numpy as np

from scipy.spatial.transform import Rotation as R
import spherical

from solvers.lifting.integration import Integrator


class AlmostTrueRotsIntegrator(Integrator):

    def __init__(self,
                 ell_max=3,
                 rots=None,
                 dtype=np.float32,
                 ):

        super().__init__(dtype=dtype, n=rots.shape[0], ell_max=ell_max, t=np.inf)

        self.rots = rots
        self.initialize_manifold()
        self.initialize_b2w()

        # Copmpute U for all the rotations
        U = np.zeros((self.n, self.ell), dtype=complex)
        wigner = spherical.Wigner(ell_max)
        quats = self.quaternions
        for i in range(self.n):
            g = quats[i]
            U[i] = wigner.D(g)

        # Compute beta
        betas = np.zeros((self.n, self.ell))
        for i in range(self.n):
            for l in range(ell_max + 1):
                for m in range(-l, l + 1):
                    for n in range(-l, l + 1):
                        if m == 0 and n == 0:
                            index = wigner.Dindex(l, m, n)
                            betas[i, index] = (2*l+1) * np.real(np.conj(U[i, index]))
                        elif m > 0 or (m == 0 and n > 0):
                            index = wigner.Dindex(l, m, n)
                            betas[i, index] = (2*l+1) * np.real(np.conj(U[i, index]))
                            indexx = wigner.Dindex(l, -m, -n)
                            betas[i, indexx] = (-1) ** (m - n) * (2*l+1) * np.imag(np.conj(U[i, indexx]))

        self.coeffs = betas

    def coeffs2weights(self, coeffs, cap_weights=True):
        weights = coeffs@self.b2w
        if cap_weights:
            weights = np.maximum(0, weights)

        reweights = np.sum(weights, axis=1)

        return weights.astype(self.dtype) / reweights[:,np.newaxis]

    def weights2coeffs(self, weights):

        return None
