import logging
import numpy as np

from solvers.lifting.problem import LiftingProblem

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

        self.cost = []
        # TODO do we want error here or just change? If so, rescaled? If so, How?

        self.vol_update = vol_update
        self.dens_update = dens_update

    def solve(self, return_result=True):
        k = 1
        # TODO redefince error
        running_error = 1.
        while running_error > self.tol and k <= self.max_it:
            logger.info(f"================================ Iteration {k} ================================")
            logger.info(f"Iteration {k} | Orientation Density update")
            self.dens_update()
            # TODO compute cost in between as well
            logger.info(f"Iteration {k} | Volume update")
            if k==1:
                fvol = self.vol_update(maxiter=5)  # TODO also make it work that we can use the result from the previous iteration + make sure that we find something so that we won't optimize too long
            else:
                fvol = self.vol_update(x0=fvol, maxiter=5)

            # update info
            cost = self.get_cost(self.problem)
            self.cost.append(cost)

            # # TODO is below still relevant?
            # error = np.sqrt(self.error_u[k - 1] ** 2 + self.error_g[k - 1] ** 2)
            # self.error_tot.append(error)
            # self.relerror_tot.append(error / self.error_tot[0])
            # running_error = self.relerror_tot[-1]
            # TODO we can look into change as criterion for convergence?


            logger.info(f"Iteration {k} | Cost = {cost}")
            # logger.info(f"Iteration {k} | Cost = {cost} | Relative Error = {running_error}")
            # logger.info(f"Iteration {k} | Relative Error u = {self.relerror_u[k-1]} | Relative Error g = {self.relerror_g[k-1]}")
            k += 1

        if return_result:
            return self.problem.vol