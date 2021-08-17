import logging
import numpy as np

from solvers.lifting.problems.inside_norm import LiftingProblem

logger = logging.getLogger(__name__)

class LiftingSolver:
    def __init__(
            self,
            problem=None,
            cost=None,
            max_it=10,
            tol=1e-4,
            vol_update=None,
            dens_update=None,
    ):
        if problem is None:
            raise RuntimeError("No problem provided")
        else:
            assert isinstance(problem, LiftingProblem)
            self.problem = problem

        if cost is None:
            raise RuntimeError("No cost function provided")
        else:
            self.get_cost = cost

        self.max_it = max_it
        self.tol = tol
        self.k = 1

        self.cost = []
        # TODO do we want error here or just change? If so, rescaled? If so, How?

        self.vol_update = vol_update
        self.dens_update = dens_update

    def solve(self, return_result=True):
        self.k = 1
        # TODO redefince error: change in volume/volume norm
        running_error = 1.
        cost = self.get_cost(self.problem)
        self.cost.append(cost)

        # TODO initialize update schemes

        # fvol = np.zeros((self.problem.L, self.problem.L, self.problem.L), dtype=self.problem.dtype)

        logger.info(f"Starting solver | Cost = {cost}")
        while running_error > self.tol and self.k <= self.max_it:
            logger.info(f"================================ Iteration {self.k} ================================")
            logger.info(f"Iteration {self.k} | Orientation Density update")

            self.dens_update(self.problem)

            cost = self.get_cost(self.problem)
            self.cost.append(cost)
            logger.info(f"Iteration {self.k} | Intermediate Cost = {cost}")
            logger.info(f"Iteration {self.k} | Volume update")
            self.vol_update(self.problem)

            # if k==1:
            #     fvol = self.vol_update(self.problem)
            # else:
            #     fvol = self.vol_update(self.problem, x0=fvol)

            # update info
            cost = self.get_cost(self.problem)
            self.cost.append(cost)

            logger.info(f"Iteration {self.k} | Cost = {cost}")
            self.k += 1

        if return_result:
            return self.problem.vol