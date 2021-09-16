import numpy as np

from solvers.lifting.problems import LiftingProblem
from solvers.lifting.problems.outside_norm import OutsideNormLiftingProblem
from solvers.lifting.problems.primal_dual_outside_norm import PrimalDualOutsideNormLiftingProblem


def l2_data_fidelity(problem):
    assert isinstance(problem, LiftingProblem)

    # Compute res
    res = problem.forward() - problem.imgs
    # Compute error
    cost = 1 / 2 * np.sum(res.asnumpy() ** 2)  # was 1 / (2 * problem.L ** 2) * np.sum(res.asnumpy() ** 2)
    return cost


def l2_data_fidelity_on(problem):
    assert isinstance(problem, OutsideNormLiftingProblem) or isinstance(problem, PrimalDualOutsideNormLiftingProblem)

    # Compute res
    integrands = problem.forward().asnumpy()
    im = problem.imgs.asnumpy()
    weights = problem.integrator.coeffs2weights(problem.rots_dcoef)

    q1 = np.sum(np.sum(weights, axis=0) * np.sum(integrands ** 2, axis=(1, 2)))
    q2 = - 2 * np.einsum("gjk,gjk", np.einsum("ig,ijk->gjk", weights, im), integrands)
    q3 = np.sum(im ** 2)

    res = q1 + q2 + q3
    # Compute error
    cost = 1 / 2 * res  # was 1 / (2 * problem.L ** 2) * res
    return cost


def l2_vol_prior(problem):
    assert isinstance(problem, LiftingProblem)

    cost = 1 / 2 * np.sum(problem.vol.asnumpy() ** 2) # was 1 / (2 * problem.L ** 3) * np.sum(problem.vol.asnumpy() ** 2)
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