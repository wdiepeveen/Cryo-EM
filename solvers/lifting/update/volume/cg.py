import logging
from functools import partial

import numpy as np
import scipy.sparse.linalg
from scipy.linalg import norm
from scipy.sparse.linalg import LinearOperator

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

import matplotlib.pyplot as plt

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


def compute_kernel(problem):
    dtype = problem.dtype

    L = problem.L
    N = problem.N
    n = problem.n

    _2L = 2 * L
    kernel = np.zeros((_2L, _2L, _2L), dtype=dtype)
    sq_filters_f = eval_filter_grid(problem, power=2)
    sq_filters_f *= problem.amplitude ** 2

    weights = np.repeat(sq_filters_f[:, :, np.newaxis], n, axis=2)

    # densities = problem.integrator.coeffs2weights(problem.rots_dcoef)
    # R = np.einsum("ij,ik->jk", densities, densities)
    # plt.imshow(R)
    # plt.colorbar()
    # plt.show()

    summed_density = np.sum(problem.integrator.coeffs2weights(problem.rots_dcoef), axis=0)
    print("summed densities = {}".format(summed_density))
    weights *= summed_density  # [np.newaxis, np.newaxis, :]

    pts_rot = rotated_grids(L, problem.integrator.rots)

    if L % 2 == 0:
        weights[0, :, :] = 0
        weights[:, 0, :] = 0

    pts_rot = m_reshape(pts_rot, (3, -1))
    weights = m_flatten(weights)

    # TODO Note: if we were to have multiple different filters,
    #  we could just sum them up as done with rho_i.
    #  In the end we just need to get weights over the integration points.
    #  We can also do this for heterogeneous setting (right?)
    kernel += (
            1
            / (N * L ** 4)  # N normalization not necessary
            * anufft(weights, pts_rot, (_2L, _2L, _2L), real=True)
    )  # TODO shouldn't we just use 1/weights?

    # Ensure symmetric kernel
    kernel[0, :, :] = 0
    kernel[:, 0, :] = 0
    kernel[:, :, 0] = 0

    logger.info("Computing non-centered Fourier Transform")
    kernel = mdim_ifftshift(kernel, range(0, 3))
    kernel_f = fft2(kernel, axes=(0, 1, 2))
    kernel_f = np.real(kernel_f)

    return FourierKernel(kernel_f, centered=False)


def src_backward(problem, basis):
    """
    Apply adjoint mapping to source

    :return: The adjoint mapping applied to the images, averaged over the whole dataset and expressed
        as coefficients of `basis`.
    """
    dtype = problem.dtype
    L = problem.L
    N = problem.N

    mean_b = np.zeros((L, L, L), dtype=dtype)

    batch_mean_b = problem.adjoint_forward(problem.imgs) / N
    mean_b += batch_mean_b.astype(dtype)

    res = basis.evaluate_t(mean_b.T)  # RCOPT
    logger.info(f"Determined adjoint mappings. Shape = {res.shape}")
    return res


def apply_kernel(vol_coeff, kernel=None, basis=None):
    """
    Applies the kernel represented by convolution
    :param vol_coeff: The volume to be convolved, stored in the basis coefficients.
    :param kernel: a Kernel object. If None, the kernel for this Estimator is used.
    :return: The result of evaluating `vol_coeff` in the given basis, convolving with the kernel given by
        kernel, and backprojecting into the basis.
    """
    assert kernel is not None
    assert basis is not None

    vol = basis.evaluate(vol_coeff).T  # RCOPT
    vol = kernel.convolve_volume(vol)  # TODO check whether we need to do some reweighting here
    vol = basis.evaluate_t(vol.T)  # RCOPT

    return vol


def conj_grad(problem, kernel, basis, b_coeff, precond_kernel=None, x0=None, regularizer=0., tol=1e-3, maxiter=20):
    dtype = problem.dtype
    n = b_coeff.shape[0]

    if regularizer > 0:
        kernel += regularizer

    operator = LinearOperator(
        (n, n), matvec=partial(apply_kernel, kernel=kernel, basis=basis), dtype=dtype
    )
    if precond_kernel is None:
        M = None
    else:
        preconditioner_kernel = precond_kernel
        if regularizer > 0:
            preconditioner_kernel += regularizer
        M = LinearOperator(
            (n, n),
            matvec=partial(apply_kernel, kernel=preconditioner_kernel, basis=basis),
            dtype=dtype,
        )

    target_residual = tol * norm(b_coeff)

    def cb(xk):
        logger.info(
            f"Delta {norm(b_coeff - apply_kernel(xk, kernel=kernel, basis=basis))} (target {target_residual})"
        )

    x, info = scipy.sparse.linalg.cg(
        operator, b_coeff, M=M, callback=cb, tol=tol, atol=0, x0=x0, maxiter=maxiter
    )

    if info != 0:
        print("Unable to converge!")
    return x


def conjugate_gradient_update(problem, basis, x0=None, regularizer=0., tol=1e-3, maxiter=20, preconditioner="circulant"):
    """
    An object representing a 2*L-by-2*L-by-2*L array containing the non-centered Fourier transform of the mean
    least-squares estimator kernel.
    Convolving a volume with this kernel is equal to projecting and backproject-ing that volume in each of the
    projection directions (with the appropriate amplitude multipliers and CTFs) and averaging over the whole
    dataset.
    Note that this is a non-centered Fourier transform, so the zero frequency is found at index 1.
    """

    dtype = problem.dtype

    if not dtype == basis.dtype:
        logger.warning(
            f"Inconsistent types in {dtype} Estimator."
            f" basis: {basis.dtype}"
        )

    logger.info("Computing kernel")
    kernel = compute_kernel(problem)

    if preconditioner == "circulant":
        logger.info("Computing Preconditioner kernel")
        precond_kernel = FourierKernel(
            1.0 / kernel.circularize(), centered=True
        )
    else:
        precond_kernel = None

    b_coeff = src_backward(problem, basis)

    est_coeff = conj_grad(problem, kernel, basis, b_coeff, precond_kernel=precond_kernel, x0=x0, regularizer=regularizer, tol=tol, maxiter=maxiter)
    est = basis.evaluate(est_coeff)
    problem.vol = Volume(est)

    return est_coeff
