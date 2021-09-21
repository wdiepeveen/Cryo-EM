

class RKHS_Kernel:
    # def __init__(self, width):
    #
    #     self.width = width

    def apply_kernel(self, omega):
        raise NotImplementedError(
            "Subclasses should implement this and return an matrix-like object"
        )

    def evaluate_gradient(self):
        # TODO think about what we want to input and output here
        raise NotImplementedError(
            "Subclasses should implement this and return an matrix-like object"
        )