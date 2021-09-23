import numpy as np


class Problem:
    def __init__(self,
                 images=None,
                 filter=None,
                 amplitude=None,
                 dtype=np.float32,
                 seed=0,
                 ):

        self.dtype = dtype

        self.images = images

        self.L = images.shape[1]
        self.N = images.shape[0]
        self.dtype = np.dtype(dtype)

        self.seed = seed

        self.filter = filter

        if amplitude is None:
            amplitude = 1.

        self.amplitude = amplitude
