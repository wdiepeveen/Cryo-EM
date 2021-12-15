import numpy as np

from pykeops.numpy import LazyTensor as LazyTensor_np

from projects.lifting_v2.src.kernels import Kernel


class Rescaled_Cosine_Kernel(Kernel):
    def __init__(self, radius=np.pi / 10, outer_quaternions=None, inner_quaternions=None, dtype=np.float32):  # , kappa=None):
        assert radius <= np.pi
        super().__init__(dtype=dtype)

        self.width = int(np.floor(np.pi / radius))  # Then we have radius <= pi/kappa

        normalisation = 2 * np.pi * self.width * (self.width ** 2 - 1) / (
                np.pi * (self.width ** 2 - 1) - self.width ** 3 * np.sin(np.pi / self.width))
        assert normalisation > 0
        self.norm = np.sqrt(normalisation)

        def construct_kernel_matrix(q1, q2, width):
            x_i = LazyTensor_np(q1[:, None, :])  # x_i.shape = (M, 1, 4)
            y_j = LazyTensor_np(q2[None, :, :])  # y_j.shape = ( 1, M, 4)
            # We can now perform large-scale computations, without memory overflows:
            distance_ij = 2 * (x_i.normalize() * y_j.normalize()).sum(-1).clamp(-1, 1).abs().acos()
            threshold_ij = (np.pi / width - distance_ij).step()
            no_thresh_kernel_ij = (width / 2 * distance_ij).cos() ** 2  # **Symbolic** (M, N) matrix

            return normalisation * threshold_ij * no_thresh_kernel_ij  # Symbolic

        if outer_quaternions is not None:
            if inner_quaternions is None:
                inner_quaternions = outer_quaternions

            self.kernel_matrix = construct_kernel_matrix(inner_quaternions, outer_quaternions, self.width)
            self.adjoint_kernel_matrix = construct_kernel_matrix(outer_quaternions, inner_quaternions, self.width)
        else:
            self.kernel_matrix = None
            self.adjoint_kernel_matrix = None

    def matrix_mult(self, vector):
        return self.kernel_matrix @ vector

    def adjoint_matrix_mult(self, vector):
        return self.adjoint_kernel_matrix @ vector

    def gradient(self, free_quaternion=None, fixed_quaternion=None):
        dist = self.manifold.dist(free_quaternion, fixed_quaternion)[:, :, :, None]
        scaling = self.norm ** 2 * self.width ** 2 * np.cos(self.width * dist / 2) * np.sinc(self.width * dist / 2)
        direction = self.manifold.log(free_quaternion, fixed_quaternion)
        return ((dist <= np.pi / self.width) * scaling * direction).squeeze()

    def kernel(self, free_quaternion=None, fixed_quaternion=None):
        dist = self.manifold.dist(free_quaternion, fixed_quaternion)[:, :, :, None]
        kernel = np.cos(self.width / 2 * dist) ** 2
        return ((dist <= np.pi / self.width) * self.norm ** 2 * kernel).squeeze()
