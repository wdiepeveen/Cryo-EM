import numpy as np
import logging

from aspire.volume import Volume
from aspire.image import Image

from projects.rkhs_lifting.src.plans import Plan
from projects.rkhs_lifting.src.plans.problems.lifting_problem1 import Lifting_Problem1
from projects.rkhs_lifting.src.plans.options.lifting_options1 import Lifting_Options1

logger = logging.getLogger(__name__)


class Lifting_Plan1(Plan):
    def __init__(self,
                 vol=None,
                 density_coeffs=None,
                 dual_coeffs=None,
                 stop=None,  # TODO here a default stopping criterion
                 stop_density_update=None,  # TODO here a default stopping criterion
                 images=None,
                 filter=None,
                 amplitude=None,
                 integrator=None,
                 dtype=np.float32,
                 seed=0,
                 ):

        if images is None:
            raise RuntimeError("No data provided")
        else:
            assert isinstance(images, Image)

        self.p = Lifting_Problem1(images=images,
                                  filter=filter,
                                  amplitude=amplitude,
                                  integrator=integrator,
                                  dtype=dtype,
                                  seed=seed)

        if vol is None:
            raise RuntimeError("No volume provided")
        else:
            assert isinstance(vol, Volume)

        if vol.dtype != self.p.dtype:
            logger.warning(
                f"{self.__class__.__name__}"
                f" vol.dtype {vol.dtype} != self.dtype {self.p.dtype}."
                " In the future this will raise an error."
            )

        # Initialize density coefficients
        if density_coeffs is None:
            density_coeffs = np.zeros((self.p.n, self.p.N), dtype=self.p.dtype)

        if dual_coeffs is None:
            dual_coeffs = np.zeros((1, self.p.N), dtype=self.p.dtype)

        self.o = Lifting_Options1(vol=vol,
                                  density_coeffs=density_coeffs,
                                  dual_coeffs=dual_coeffs,
                                  stop=stop,
                                  stop_density_update=stop_density_update,
                                  )

    def get_cost(self):
        k = 1  # TODO get from options and hard-code this here
