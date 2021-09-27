import numpy as np

class RKHS_Kernel:

    def __init__(self, dtype=np.float32):
        self.dtype = dtype

        self.norm = None
        self.kernel_matrix = None

    def matrix_mult(self, vector):
        raise NotImplementedError(
            "Subclasses should implement this and return an matrix-like object"
        )

    def evaluate_gradient(self):
        # TODO think about what we want to input and output here
        raise NotImplementedError(
            "Subclasses should implement this and return an matrix-like object"
        )