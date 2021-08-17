import numpy as np
import logging

from scipy.fftpack import fft2

from aspire.nufft import anufft
from aspire.reconstruction import Estimator, FourierKernel
from aspire.utils.fft import mdim_ifftshift
from aspire.utils.matlab_compat import m_flatten, m_reshape
from aspire.volume import rotated_grids

from aspire import config
from aspire.reconstruction.kernel import FourierKernel
from aspire.utils.coor_trans import grid_2d
from aspire.volume import Volume

from solvers.lifting.problems.outside_norm import OutsideNormLiftingProblem
from solvers.lifting.problems.primal_dual_outside_norm import PrimalDualOutsideNormLiftingProblem


logger = logging.getLogger(__name__)


def eval_filter_grid(problem, power=1):
    dtype = problem.dtype
    L = problem.L

    grid2d = grid_2d(L, dtype=dtype)
    omega = np.pi * np.vstack((grid2d["x"].flatten(), grid2d["y"].flatten()))

    filter_values = problem.filter.evaluate(omega)
    if power != 1:
        filter_values **= power

    h = np.reshape(filter_values, grid2d["x"].shape)

    return h


def exact_refinement(problem, rots, sq_sigma=1., regularizer=1.):
    assert isinstance(problem, OutsideNormLiftingProblem) or isinstance(problem, PrimalDualOutsideNormLiftingProblem)

    dtype = problem.dtype

    L = problem.L
    N = problem.N

    # compute adjoint forward map of images
    src = problem.adjoint_forward(problem.imgs)

    # compute kernel in fourier domain

    _2L = 2 * L
    kernel = np.zeros((_2L, _2L, _2L), dtype=dtype)
    sq_filters_f = eval_filter_grid(problem, power=2)
    sq_filters_f *= problem.amplitude ** 2

    weights = np.repeat(sq_filters_f[:, :, np.newaxis], N, axis=2)

    # summed_density = np.sum(problem.integrator.coeffs2weights(problem.rots_dcoef), axis=0)
    # print("summed densities = {}".format(summed_density))
    # weights *= summed_density  # [np.newaxis, np.newaxis, :]

    pts_rot = rotated_grids(L, rots)
    # pts_rot = np.moveaxis(pts_rot, 1, 2)  # TODO do we need this?
    pts_rot = m_reshape(pts_rot, (3, -1))

    if L % 2 == 0:
        weights[0, :, :] = 0
        weights[:, 0, :] = 0

    weights = m_flatten(weights)

    kernel += (
            1
            / (L ** 4)  # TODO check whether scaling is correct like this
            * anufft(weights, pts_rot, (_2L, _2L, _2L), real=True)
    )

    # Ensure symmetric kernel
    kernel[0, :, :] = 0
    kernel[:, 0, :] = 0
    kernel[:, :, 0] = 0

    logger.info("Computing non-centered Fourier Transform")
    kernel = mdim_ifftshift(kernel, range(0, 3))
    kernel_f = fft2(kernel, axes=(0, 1, 2))
    kernel_f = np.real(kernel_f)

    f_kernel = FourierKernel(kernel_f, centered=False)
    f_kernel += regularizer * sq_sigma

    f_kernel = FourierKernel(
            1.0 / f_kernel.kernel, centered=False
        )

    # apply kernel
    vol = np.real(f_kernel.convolve_volume(src.T)).astype(dtype)  # TODO works, but still not entirely sure why we need to transpose here

    return Volume(vol)