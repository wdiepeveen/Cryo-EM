

class RKHS_Kernel:

    def matrix_mult(self, vector):
        raise NotImplementedError(
            "Subclasses should implement this and return an matrix-like object"
        )

    def evaluate_gradient(self):
        # TODO think about what we want to input and output here
        raise NotImplementedError(
            "Subclasses should implement this and return an matrix-like object"
        )