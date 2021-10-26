import logging

from projects.rkhs_lifting.src.plans.options import Options

logger = logging.getLogger(__name__)


class Refinement_Options1(Options):
    def __init__(self,
                 squared_noise_level=None,
                 stop=None,
                 stop_rots_gd=None,
                 rots_batch_size=None,
                 ):
        super().__init__(stop=stop)

        self.squared_noise_level = squared_noise_level
        self.stop_rots_gd = stop_rots_gd
        self.rots_batch_size = rots_batch_size

