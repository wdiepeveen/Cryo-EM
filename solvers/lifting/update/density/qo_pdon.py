import numpy as np
import osqp

from scipy import sparse

import logging

from solvers.lifting.problems.primal_dual_outside_norm import PrimalDualOutsideNormLiftingProblem

logger = logging.getLogger(__name__)


def primal_dual_quadratic_optimisation_update(problem, sq_sigma=1., regularizer=1.):
    assert isinstance(problem, PrimalDualOutsideNormLiftingProblem)

    dtype = problem.dtype

    ell = problem.ell
    n = problem.n

    L = problem.L
    N = problem.N

    # Compute Q:
    logger.info("Computing P")
    P = regularizer * np.eye(ell)

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
    P = sparse.csc_matrix(P)
    qq = qq.astype(np.float64)

    H = np.eye(1, ell)
    G = problem.integrator.b2w.T.astype(np.float64)
    A = np.vstack([H, G])
    A = sparse.csc_matrix(A)

    l = np.eye(1, n+1)[0]  # TODO use A shape instead of n
    u = np.eye(1, n+1)[0]  # TODO use A shape instead of n
    u[1:] = np.inf

    # Initialize solver
    m = osqp.OSQP()

    # Loop over all images
    logger.info("Solving for betas")
    beta = np.zeros((N, ell))
    dual_beta = np.zeros((N, n+1))  # TODO use A shape instead of n
    for i in range(N):
        q = qq[i]
        if i==0:
            # Setup
            m.setup(P=P, q=q, A=A, l=l, u=u)  # was max_iter=150
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
