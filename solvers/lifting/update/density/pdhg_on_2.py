import numpy as np
import osqp

from scipy import sparse

import logging

from solvers.lifting.problems.primal_dual_outside_norm import PrimalDualOutsideNormLiftingProblem
from solvers.lifting.update.density.solvers.pdhg import PDHG

logger = logging.getLogger(__name__)

# TODO make this into a class that has an initialize function and a solve function. We can then just use those routines
#  and then we don't have to compute (and send to device) the operators A


def primal_dual_hybrid_gradient_update(problem, sq_sigma=1., regularizer=1.):
    assert isinstance(problem, PrimalDualOutsideNormLiftingProblem)

    dtype = problem.dtype

    ell = problem.ell
    n = problem.n

    L = problem.L
    N = problem.N

    # Compute q:
    logger.info("Computing qs")
    integrands = problem.forward().asnumpy()
    im = problem.imgs.asnumpy()

    q1 = np.repeat(np.sum(integrands**2, axis=(1, 2))[np.newaxis, :], N, axis=0)
    q2 = - 2 * np.einsum("ijk,gjk->ig", im, integrands)
    q3 = np.repeat(np.sum(im**2, axis=(1, 2))[:, np.newaxis], n, axis=1)
    qs = (q1 + q2 + q3) / (sq_sigma * L**2)

    qq = qs @ problem.integrator.b2w.T

    print("qq's = {}".format(qq[0,0:5]))

    # Update:
    # P = sparse.csc_matrix(P)
    # qq = qq.astype(np.float64)

    # H = np.eye(1, ell).astype(dtype)
    # if isinstance(problem.integrator.b2w, sparse.csr.csr_matrix):
    #     print("Sparse b2w matrix recognized")
    #     # H = sparse.csc_matrix(H)
    #     # G = problem.integrator.b2w.T  #.astype(np.float64)
    #     # A = sparse.vstack([H, G], format="csc")
    #     A = np.vstack([H, problem.integrator.b2w.T])
    # else:
    #     G = problem.integrator.b2w.T  #.astype(np.float64)
    #     A = np.vstack([H, G])
    #     # A = sparse.csc_matrix(A)

    A = problem.integrator.b2w.T[:,1:]

    # Loop over all images
    logger.info("Solving for betas")  # TODO shouldn't this just already be in problem?
    beta = problem.rots_dcoef[:,1:]
    # TODO we don't need the first dual variable anymore
    dual_beta = problem.dual_rots_dcoef[:,1:]

    # beta = np.zeros((N, ell))
    # dual_beta = np.zeros((N, n + 1))

    def primal_prox(primals, sigma):
        # print("primals before = {}".format(primals[0,1:10]))
        result = 1/(1 + regularizer * sigma) * (primals - sigma * qq[:,1:])
        # print("primals after = {}".format(result[0,1:10]))
        return result

    def dual_prox(duals, tau):
        # print("duals before = {}".format(duals[0,1:10]))
        result = duals + tau/n * np.ones(duals.shape)
        result *= (result <= 0)
        # print("duals after = {}".format(result[0,1:10]))
        return result

    # TODO maybe reshape the whole thing so that it is a bit clearer what we are actually doing?
    def block_operator(primals):
        return np.einsum("ij,kj->ik", primals, A)

    def adjoint_block_operator(duals):
        return np.einsum("ij,kj->ik", duals, A.T)

    # TODO Compute operator norm of matrix to estimate sigma and tau

    solver = PDHG(primal_prox,dual_prox,block_operator,adjoint_block_operator,beta,dual_beta,sigma=1/2, tau=1/2, gamma=0.) #regularizer)

    solver.solve()
    beta = solver.x
    dual_beta = solver.y

    # for i in range(N):
    #     q = qq[i]
    #     if i==0:
    #         # Setup
    #         m.setup(P=P, q=q, A=A, l=l, u=u)  # was max_iter=150
    #         logger.info("Setup complete")
    #     else:
    #         # Update
    #         m.update(q=q)
    #
    #     x0 = problem.rots_dcoef[i]
    #     y0 = problem.dual_rots_dcoef[i]
    #     m.warm_start(x=x0, y=y0)
    #
    #     results = m.solve()
    #     beta[i] = results.x
    #     dual_beta[i] = results.y
    #
    #     if (i+1) % int(N/10) == 0:
    #         logger.info("Density update at {}%".format(int((i+1)/N*100)))

    problem.rots_dcoef[:,1:] = beta.astype(dtype)
    problem.dual_rots_dcoef[:,1:] = dual_beta.astype(dtype)

    return solver.normalized_errors
