import numpy as np
import logging

from projects.rkhs_lifting.src.plans.problems import Problem

logger = logging.getLogger(__name__)


class Refinement_Problem1(Problem):
    def __init__(self,
                 images=None,
                 filter=None,
                 amplitude=None,
                 kernel=None,
                 integrator=None,
                 dtype=np.float32,
                 seed=0,
                 ):
        """
               A Cryo-EM refinement problem
               Other than the base class attributes, it has:
               :param imgs: A n-by-L-by-L array of noisy projection images (should not be confused with the .image objects)
               """

        super().__init__(images=images, filter=filter, amplitude=amplitude, dtype=dtype, seed=seed)

        self.kernel = kernel
        self.integrator = integrator
        self.n = self.integrator.n


