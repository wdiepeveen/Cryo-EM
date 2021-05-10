import numpy as np
import quadprog

import logging

from solvers.lifting.problems.outside_norm import OutsideNormLiftingProblem

logger = logging.getLogger(__name__)


def quadprog_solve_qp(Q, q, G=None, h=None, A=None, b=None):
    qp_G = .5 * (Q + Q.T)  # make sure P is symmetric
    qp_a = -q
    if A is not None:
        qp_C = -np.vstack([A, G]).T
        qp_b = -np.hstack([b, h])
        meq = A.shape[0]
    else:  # no equality constraint
        qp_C = -G.T
        qp_b = -h
        meq = 0
    return quadprog.solve_qp(qp_G, qp_a, qp_C, qp_b, meq)[0]


def quadratic_optimisation_update(problem, sq_sigma=1., regularizer=1.):
    assert isinstance(problem, OutsideNormLiftingProblem)

    dtype = problem.dtype

    ell = problem.ell
    n = problem.n

    L = problem.L
    N = problem.N

    # Compute Q:
    logger.info("Computing Q")
    Q = regularizer * np.eye(ell)

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

    G = - problem.integrator.b2w.T.astype(np.float64)
    h = np.zeros((n,))
    A = np.eye(1, ell)
    b = np.ones((1,))

    # Loop over all images
    logger.info("Solving for betas")
    beta = np.zeros((N, ell))
    for i in range(N):
        if (i+1) % int(N/10) == 0:
            logger.info("Density update at {}%".format(int((i+1)/N*100)))

        q = qq[i]
        beta[i] = quadprog_solve_qp(Q, q, G=G, h=h, A=A, b=b)

    problem.rots_dcoef = beta.astype(dtype)
