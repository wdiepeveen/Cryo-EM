import numpy as np

from pykeops.numpy import LazyTensor as LazyTensor_np

from projects.rkhs_lifting.src.kernels import RKHS_Kernel


class Rescaled_Cosine_Kernel(RKHS_Kernel):
    def __init__(self, quaternions=None, radius=np.pi / 10, dtype=np.float32):  # , kappa=None):
        assert radius < np.pi
        super().__init__(dtype=dtype)

        # Use separation distance between grid points to find suitable kappa
        self.width = int(np.floor(np.pi / radius))  # Then we have radius <= pi/kappa

        x_i = LazyTensor_np(quaternions[:, None, :])  # x_i.shape = (M, 1, 4)
        y_j = LazyTensor_np(quaternions[None, :, :])  # y_j.shape = ( 1, M, 4)

        # We can now perform large-scale computations, without memory overflows:
        distance_ij = 2 * (x_i.normalize() * y_j.normalize()).sum(-1).clamp(-1, 1).abs().acos()
        threshold_ij = (np.pi / self.width - distance_ij).step()
        no_thresh_kernel_ij = (self.width / 2 * distance_ij).cos() ** 2  # **Symbolic** (M, N) matrix
        normalisation = 2 * np.pi * self.width * (self.width ** 2 - 1) / (
                    np.pi * (self.width ** 2 - 1) - self.width ** 3 * np.sin(np.pi / self.width))
        assert normalisation > 0
        kernel_ij = normalisation * threshold_ij * no_thresh_kernel_ij  # Symbolic

        self.norm = np.sqrt(normalisation)
        self.kernel_matrix = kernel_ij

    def matrix_mult(self, vector):
        return self.kernel_matrix @ vector

    def gradient(self, free_quaternion=None, fixed_quaternion=None):
        dist = self.manifold.dist(free_quaternion, fixed_quaternion)
        scaling = self.norm**2 * self.width**2 * np.cos(self.width * dist/2) * np.sinc(self.width * dist/2)
        direction = self.manifold.log(free_quaternion, fixed_quaternion)
        return (dist <= np.pi/self.width) * scaling * direction
