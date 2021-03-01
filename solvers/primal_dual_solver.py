import logging
import numpy as np

from aspire.image.image import Image
from aspire.reconstruction.estimator import Estimator
from aspire.volume import Volume

from pymanopt.manifolds import Rotations

from scipy.linalg import norm

# Own lib
from problems.primal_dual_problem import PrimalDualProblem
from solvers import Solver

logger = logging.getLogger(__name__)

# TODO make Solver object instead of an Estimator
class PrimalDualSolver(Estimator):
    def __init__(
            self,
            problem=None,
            cost=None,
            max_it=200,
            tol=1e-4,
            # u params
            gradient_u=None,
            alpha=1 / 2,
            # g params
            xi=None,
            prox_primal_g=None,
            prox_dual_g=None,
            differential_g=None,
            adjoint_g=None,
            sigma=1 / 2,
            tau=1 / 2,
            gamma=0.2
    ):
        if problem is None:
            raise RuntimeError("No problem provided")
        else:
            assert isinstance(problem, PrimalDualProblem)
            self.problem = problem
        # TODO super init
        if cost is None:
            raise RuntimeError("No cost function provided")
        else:
            self.get_cost = cost

        self.max_it = max_it
        self.tol = tol

        self.cost = []
        self.error_u = []
        self.relerror_u = []
        self.error_g = []
        self.relerror_g = []
        self.error_tot = []
        self.relerror_tot = []

        # u parameters
        if gradient_u is None:
            raise RuntimeError("No gradient method for u provided")
        else:
            self.gradient_u = gradient_u

        self.alpha = alpha

        # g parameters
        if xi is None:
            self.xi = np.zeros((self.problem.n, self.problem.L, self.problem.L), dtype=self.problem.dtype)
        else:
            assert isinstance(xi, Image)
            assert xi.dtype == self.problem.dtype
            self.xi = xi

        self.xi_ = self.xi

        if prox_primal_g is None:
            raise RuntimeError("No primal proximal map method for g provided")
        else:
            self.prox_primal_g = prox_primal_g

        if prox_dual_g is None:
            raise RuntimeError("No dual proximal map method for g provided")
        else:
            self.prox_dual_g = prox_dual_g

        if differential_g is None:
            raise RuntimeError("No differential method for g provided")
        else:
            self.differential_g = differential_g

        if adjoint_g is None:
            raise RuntimeError("No adjoint map method for g provided")
        else:
            self.adjoint_g = adjoint_g

        self.sigma = sigma
        self.tau = tau
        self.gamma = gamma

    def solve(self,return_result=True):
        k = 1
        running_error = 1.
        while running_error > self.tol and k <= self.max_it:
            logger.info(f"========= Iteration {k} =========")
            logger.info(f"Iteration {k} | descent step u")
            self.gradient_descent_step_u()
            logger.info(f"Iteration {k} | primal-dual step g")
            self.primal_dual_step_g()

            # update info
            cost = self.get_cost(self.problem)
            self.cost.append(cost)
            error = np.sqrt(self.error_u[k - 1] ** 2 + self.error_g[k - 1] ** 2)
            self.error_tot.append(error)
            self.relerror_tot.append(error / self.error_tot[0])
            running_error = self.relerror_tot[-1]
            logger.info(f"Iteration {k} | Cost = {cost} | Relative Error = {running_error}")
            logger.info(f"Iteration {k} | Relative Error u = {self.relerror_u[k-1]} | Relative Error g = {self.relerror_g[k-1]}")
            k += 1

        if return_result:
            return self.problem.vols

    def gradient_descent_step_u(self):
        gradient = self.alpha * self.gradient_u(self.problem).asnumpy()[0]
        u = self.problem.vols.asnumpy()[0] - gradient
        error = norm(gradient)
        self.error_u.append(error)
        self.relerror_u.append(error / self.error_u[0])
        self.problem.vols = Volume(u)

    def primal_dual_step_g(self):
        manifold = Rotations(3, self.problem.n)
        m = self.problem.rots
        old_xi = self.xi
        # TODO print first items from all inputs
        print("xi_={}".format(self.xi_[0]))
        print("D Lambda[xi_] = {}".format(self.adjoint_g(self.problem, self.xi_)[0]))
        # primal update
        new_rots = self.prox_primal_g(self.problem, self.sigma,
                                               manifold.exp(m, -self.sigma * \
                                                            self.adjoint_g(self.problem, self.xi_)
                                                            )
                                               )
        primal_error = manifold.dist(m, new_rots)
        # dual update
        self.xi = self.prox_dual_g(self.problem, self.tau,
                                   self.xi.data + self.tau * \
                                   self.differential_g(self.problem,
                                                        manifold.log(m, new_rots))
                                   )
        dual_error = norm(self.xi.data - old_xi.data)

        theta = 1 / (np.sqrt(1 + 2 * self.gamma * self.sigma))
        self.sigma = self.sigma * theta
        self.tau = self.tau / theta

        self.problem.rots = new_rots
        self.xi_ = Image(self.xi.data + theta * (self.xi.data - old_xi.data))
        error = np.sqrt(primal_error ** 2 + dual_error ** 2)
        self.error_g.append(error)
        self.relerror_g.append(error / self.error_g[0])
