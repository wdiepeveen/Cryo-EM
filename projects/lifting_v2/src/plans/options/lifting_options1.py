import logging

from projects.lifting_v2.src.plans.options import Options

logger = logging.getLogger(__name__)


class Lifting_Options1(Options):
    def __init__(self,
                 vol=None,
                 squared_noise_level=None,
                 density_coeffs=None,
                 max_iter=None,
                 rots_batch_size=None,
                 ):
        super().__init__(max_iter=max_iter)
        self.vol = vol
        self.squared_noise_level = squared_noise_level

        self.density_coeffs = density_coeffs

        self.rots_batch_size = rots_batch_size

    def get_solver_result(self):
        return self.vol
