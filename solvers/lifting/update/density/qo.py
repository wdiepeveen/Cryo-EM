import numpy as np

import quadprog


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


def quadratic_optimisation_update(problem, sq_sigma=1., reg1=1., reg2=1.):

    dtype = problem.dtype

    ell = problem.ell
    n = problem.n

    L = problem.L
    N = problem.N

    # Compute Q:
    integrands = problem.integrands_forward().asnumpy()
    B = np.einsum("ijk,ljk->il", integrands, integrands) / (sq_sigma * L**2)
    Q = problem.integrator.b2w @ B @ problem.integrator.b2w.T
    Q += reg1 * np.eye(ell)

    # Compute q:
    integrands = problem.integrands_forward().asnumpy()
    im = problem.imgs.asnumpy()
    qs = - np.einsum("ijk,ljk->il", im, integrands) / (sq_sigma * L**2)
    # if problem.rots_prior_integrands is not None:
    #     qs += reg2 * problem.rots_prior_integrands

    qq = qs @ problem.integrator.b2w.T

    # Update:
    Q = Q.astype(np.float64)
    qq = qq.astype(np.float64)

    G = - problem.integrator.b2w.T.astype(np.float64)
    h = np.zeros((n,))
    A = np.eye(1, ell)
    b = np.ones((1,))

    # Loop over all images
    beta = np.zeros((N, ell))
    for i in range(N):
        q = qq[i]
        beta[i] = quadprog_solve_qp(Q, q, G=G, h=h, A=A, b=b)

    problem.rots_dcoef = beta.astype(dtype)
