import numpy as np

from pykeops.numpy import LazyTensor as LazyTensor_np

from solvers.lifting.integration.rkhs.kernels import RKHS_Kernel


class Rescaled_Cosine_Kernel(RKHS_Kernel):
    def __init__(self, quaternions, radius):  # , kappa=None):

        # Use separation distance between grid points to find suitable kappa
        width = int(np.floor(np.pi/radius))  # Then we have radius <= pi/kappa

        super().__init__(width)

        x_i = LazyTensor_np(quaternions[:, None, :])  # x_i.shape = (M, 1, 4)
        y_j = LazyTensor_np(quaternions[None, :, :])  # y_j.shape = ( 1, M, 4)

        # We can now perform large-scale computations, without memory overflows:
        distance_ij = 2 * (x_i.normalize() * y_j.normalize()).sum(-1).clamp(-1, 1).abs().acos()
        threshold_ij = (np.pi / self.width - distance_ij).step()
        no_thresh_kernel_ij = (self.width / 2 * distance_ij).cos() ** 2  # **Symbolic** (M, N) matrix
        kernel_ij = threshold_ij * no_thresh_kernel_ij  # Symbolic
        print("K_ij = {}".format(kernel_ij))
        print("K_ij[1,2] = {}".format(kernel_ij[0,:10]))

        self.kernel_matrix = kernel_ij

    def apply_kernel(self, vector):
        return vector @ self.kernel_matrix  # Assuming here that we are dealing with a row vector