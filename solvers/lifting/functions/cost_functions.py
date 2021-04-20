import numpy as np

from solvers.lifting.problem import LiftingProblem


def l2_data_fidelity(problem):
    assert isinstance(problem, LiftingProblem)

    # Compute res
    res = problem.forward() - problem.imgs
    # Compute error
    cost = 1 / (2 * problem.L ** 2) * np.sum(res.asnumpy() ** 2)
    return cost


def l2_vol_prior(problem):
    assert isinstance(problem, LiftingProblem)

    cost = 1 / (2 * problem.L ** 3) * np.sum(problem.vol.asnumpy() ** 2)
    return cost


def l2_dens_prior(problem):
    assert isinstance(problem, LiftingProblem)

    cost = 1 / 2 * np.sum(problem.rots_dcoef ** 2)
    return cost


def integral_dens_prior(problem):
    assert isinstance(problem, LiftingProblem)
    assert problem.rots_prior_integrands is not None

    weights = problem.integrator.coeffs2weights(problem.rots_dcoef)
    integrands = problem.rots_prior_integrands

    cost = np.sum(weights * integrands)
    return cost