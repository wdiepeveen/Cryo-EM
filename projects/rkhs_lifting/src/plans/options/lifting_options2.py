import logging

from projects.rkhs_lifting.src.plans.options import Options

logger = logging.getLogger(__name__)


class Lifting_Options2(Options):
    def __init__(self,
                 vol=None,
                 squared_noise_level=None,
                 density_coeffs=None,
                 drs_coeffs=None,
                 stop=None,
                 stop_density_update=None,
                 batch_size=None,
                 ):
        super().__init__(stop=stop)
        self.vol = vol
        self.squared_noise_level = squared_noise_level

        self.density_coeffs = density_coeffs
        self.drs_coeffs = drs_coeffs

        self.stop_density_update = stop_density_update

        self.batch_size = batch_size

    def get_solver_result(self):
        return self.vol
