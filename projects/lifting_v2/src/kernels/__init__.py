import numpy as np

from projects.lifting_v2.src.manifolds.so3 import SO3

class Kernel:

    def __init__(self, dtype=np.float32):
        self.dtype = dtype
        self.manifold = SO3()

        self.norm = None
        self.kernel_matrix = None
        self.adjoint_kernel_matrix = None

    def matrix_mult(self, vector):
        raise NotImplementedError(
            "Subclasses should implement this and return an matrix-like object"
        )

    def adjoint_matrix_mult(self, vector):
        raise NotImplementedError(
            "Subclasses should implement this and return an matrix-like object"
        )

    def gradient(self, free_quaternion=None, fixed_quaternion=None):
        # TODO think about what we want to input and output here
        raise NotImplementedError(
            "Subclasses should implement this and return an matrix-like object"
        )
