import logging

from projects.rkhs_lifting.src.plans.options import Options

logger = logging.getLogger(__name__)


class Lifting_Options1(Options):
    def __init__(self,
                 vol=None,
                 squared_noise_level=None,
                 density_coeffs=None,
                 dual_coeffs=None,
                 stop=None,
                 stop_density_update=None,
                 rots_batch_size=None,
                 ):
        super().__init__(stop=stop)
        self.vol = vol
        self.squared_noise_level = squared_noise_level

        self.density_coeffs = density_coeffs
        self.dual_coeffs = dual_coeffs

        self.stop_density_update = stop_density_update

        self.rots_batch_size = rots_batch_size

    def get_solver_result(self):
        return self.vol
