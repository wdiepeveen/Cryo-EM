import numpy as np
import osqp

import logging

from solvers.lifting.problems.primal_dual_outside_norm import PrimalDualOutsideNormLiftingProblem

logger = logging.getLogger(__name__)


# def quadprog_solve_qp(Q, q, G=None, h=None, A=None, b=None):
#     qp_G = .5 * (Q + Q.T)  # make sure P is symmetric
#     qp_a = -q
#     if A is not None:
#         qp_C = -np.vstack([A, G]).T
#         qp_b = -np.hstack([b, h])
#         meq = A.shape[0]
#     else:  # no equality constraint
#         qp_C = -G.T
#         qp_b = -h
#         meq = 0
#     return quadprog.solve_qp(qp_G, qp_a, qp_C, qp_b, meq)[0]


def quadratic_optimisation_update(problem, sq_sigma=1., regularizer=1.):
    assert isinstance(problem, PrimalDualOutsideNormLiftingProblem)

    dtype = problem.dtype

    ell = problem.ell
    n = problem.n

    L = problem.L
    N = problem.N

    # Compute Q:
    logger.info("Computing P")
    P = regularizer * np.eye(ell) # TODO to scipy CSC matrix

    # Compute q:
    logger.info("Computing qs")
    integrands = problem.forward().asnumpy()
    im = problem.imgs.asnumpy()

    q1 = np.repeat(np.sum(integrands**2, axis=(1, 2))[np.newaxis, :], N, axis=0)
    q2 = - 2 * np.einsum("ijk,gjk->ig", im, integrands)
    q3 = np.repeat(np.sum(im**2, axis=(1, 2))[:, np.newaxis], n, axis=1)
    qs = (q1 + q2 + q3) / (sq_sigma * L**2)

    qq = qs @ problem.integrator.b2w.T

    # Update:
    Q = Q.astype(np.float64)
    qq = qq.astype(np.float64)

    H = np.eye(1, ell)
    G = problem.integrator.b2w.T.astype(np.float64)
    A = np.vstack([H, G])  # TODO to scipy CSC matrix

    l = u = np.eye(1, n)[0]
    u[1:] = np.inf

    # Initialize solver
    m = osqp.OSQP()

    # Loop over all images
    logger.info("Solving for betas")
    beta = np.zeros((N, ell))
    dual_beta = np.zeros((N, n))
    for i in range(N):
        q = qq[i]
        if i==0:
            # Setup
            m.setup(P=P, q=q, A=A, l=l, u=u)
        else:
            # Update
            m.update(q=q)

        x0 = problem.rots_dcoef[i]
        y0 = problem.dual_rots_dcoef[i]
        m.warm_start(x=x0, y=y0)

        results = m.solve()
        beta[i] = results.x
        dual_beta[i] = results.y

        if (i+1) % int(N/10) == 0:
            logger.info("Density update at {}%".format(int((i+1)/N*100)))

    problem.rots_dcoef = beta.astype(dtype)
    problem.dual_rots_dcoef = dual_beta.astype(dtype)
