import numpy as np
import logging

from projects.lifting_v2.src.integrators import RKHS_Density_Integrator
from projects.lifting_v2.src.plans.problems import Problem

logger = logging.getLogger(__name__)


class Lifting_Problem2(Problem):
    def __init__(self,
                 images=None,
                 filter=None,
                 amplitude=None,
                 integrator=None,
                 volume_reg_param=None,
                 rots_density_reg_param=None,
                 dtype=np.float32,
                 seed=0,
                 ):
        """
               A Cryo-EM lifting problem
               Other than the base class attributes, it has:
               :param imgs: A n-by-L-by-L array of noisy projection images (should not be confused with the .image objects)
               """

        super().__init__(images=images, filter=filter, amplitude=amplitude, dtype=dtype, seed=seed)

        if not issubclass(type(integrator), RKHS_Density_Integrator):
            raise RuntimeError("integrator is not an Integrator object")
        self.integrator = integrator
        self.n = self.integrator.n

        self.volume_reg_param = volume_reg_param
        self.rots_density_reg_param = rots_density_reg_param

        # TODO here maybe some functions such as cost, proxes


