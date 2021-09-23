import logging

from projects.rkhs_lifting.src.plans.options import Options

logger = logging.getLogger(__name__)


class Lifting_Options1(Options):
    def __init__(self,
                 vol=None,
                 density_coeffs=None,
                 dual_coeffs=None,
                 stop=None,
                 stop_density_update=None,
                 ):
        super().__init__(stop=stop)
        self.vol = vol

        self.density_coeffs = density_coeffs
        self.dual_coeffs = dual_coeffs

        self.stop_density_update = stop_density_update

    def get_solver_result(self):
        return self.vol
