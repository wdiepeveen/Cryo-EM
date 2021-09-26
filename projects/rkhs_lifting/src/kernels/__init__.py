import numpy as np

class RKHS_Kernel:

    def __init__(self, dtype=np.float32):
        if dtype == np.float32:
            dtype_str = "float32"
        elif dtype == np.float64:
            dtype_str = "float64"
        else:
            raise NotImplementedError(
                "Choose either np.float32 or np.float64 as dtype"
            )
        self.dtype = dtype_str

    def matrix_mult(self, vector):
        raise NotImplementedError(
            "Subclasses should implement this and return an matrix-like object"
        )

    def evaluate_gradient(self):
        # TODO think about what we want to input and output here
        raise NotImplementedError(
            "Subclasses should implement this and return an matrix-like object"
        )