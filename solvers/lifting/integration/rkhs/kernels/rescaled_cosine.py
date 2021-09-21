import numpy as np

from pykeops.numpy import LazyTensor as LazyTensor_np

from solvers.lifting.integration.rkhs.kernels import RKHS_Kernel


class Rescaled_Cosine_Kernel(RKHS_Kernel):
    def __init__(self, quats, radius):  # , kappa=None):

        # Use separation distance between grid points to find suitable kappa
        kappa = int(np.floor(np.pi/radius))  # Then we have radius <= pi/kappa

        super().__init__(kappa)

        x_i = LazyTensor_np(quats[:, None, :])  # x_i.shape = (M, 1, 4)
        y_j = LazyTensor_np(quats[None, :, :])  # y_j.shape = ( 1, M, 4)

        # We can now perform large-scale computations, without memory overflows:
        Omega_ij = 2 * (x_i.normalize() * y_j.normalize()).sum(-1).clamp(-1, 1).abs().acos()
        S_ij = (np.pi / self.width - Omega_ij).step()
        D_ij = (self.width / 2 * Omega_ij).cos() ** 2  # **Symbolic** (M, N) matrix
        K_ij = S_ij * D_ij  # Symbolic
        print("K_ij = {}".format(K_ij))
        print("K_ij[1,2] = {}".format(K_ij[:10]))

        self.kernel = K_ij

    def apply_kernel(self, vector):
        return self.kernel.matvecmult(vector)
        # return self.kernel.vecmatmult(vector)