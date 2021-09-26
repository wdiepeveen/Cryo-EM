import numpy as np
from scipy.special import gammaln

from pykeops.numpy import LazyTensor as LazyTensor_np

from projects.rkhs_lifting.src.kernels import RKHS_Kernel


class Vallee_Poussin_Kernel(RKHS_Kernel):  # TODO Test!
    def __init__(self, quaternions, kappa, dtype=np.float32):
        super().__init__(dtype=dtype)

        x_i = LazyTensor_np(quaternions[:, None, :]).dtype(self.dtype)  # x_i.shape = (M, 1, 4)
        y_j = LazyTensor_np(quaternions[None, :, :]).dtype(self.dtype)  # y_j.shape = ( 1, M, 4)

        # We can now perform large-scale computations, without memory overflows:
        distance_ij = 2 * (x_i.normalize() * y_j.normalize()).sum(-1).clamp(-1, 1).abs().acos()
        normalisation = np.sqrt(np.pi) * np.exp(gammaln(kappa + 2) - gammaln(kappa + 1 / 2))
        kernel_ij = normalisation * (distance_ij / 2).cos() ** (2*kappa)  # **Symbolic** (M, N) matrix
        # print("K_ij = {}".format(kernel_ij))

        self.kernel_matrix = kernel_ij

    def matrix_mult(self, vector):
        return self.kernel_matrix @ vector
