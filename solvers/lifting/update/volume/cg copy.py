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

from solvers.lifting.update.volume import VolumeUpdate

logger = logging.getLogger(__name__)


class LeastSquaresCGUpdate(VolumeUpdate):
    def __init__(self, problem, basis, batch_size=512, preconditioner="circulant"):  # TODO batchsize is unused right now

        super().__init__(problem=problem)

        self.basis = basis

        self.batch_size = batch_size
        self.preconditioner = preconditioner

        if not self.dtype == self.basis.dtype:
            logger.warning(
                f"Inconsistent types in {self.dtype} Estimator."
                f" basis: {self.basis.dtype}"
            )

        """
        An object representing a 2*L-by-2*L-by-2*L array containing the non-centered Fourier transform of the mean
        least-squares estimator kernel.
        Convolving a volume with this kernel is equal to projecting and backproject-ing that volume in each of the
        projection directions (with the appropriate amplitude multipliers and CTFs) and averaging over the whole
        dataset.
        Note that this is a non-centered Fourier transform, so the zero frequency is found at index 1.
        """

    def __getattr__(self, name):
        # TODO?
        """Lazy attributes instantiated on first-access"""

        if name == "kernel":
            logger.info("Computing kernel")
            kernel = self.kernel = self.compute_kernel()
            return kernel

        elif name == "precond_kernel":
            if self.preconditioner == "circulant":
                logger.info("Computing Preconditioner kernel")
                precond_kernel = self.precond_kernel = FourierKernel(
                    1.0 / self.kernel.circularize(), centered=True
                )
            else:
                precond_kernel = self.precond_kernel = None
            return precond_kernel
        # TODO not sure what line below means
        return super(LeastSquaresCGUpdate, self).__getattr__(name)

    def compute_kernel(self):
        _2L = 2 * self.L
        kernel = np.zeros((_2L, _2L, _2L), dtype=self.dtype)
        sq_filters_f = self.eval_filter_grid(power=2)
        sq_filters_f *= self.problem.amplitude ** 2

        weights = np.repeat(sq_filters_f[:, :, np.newaxis], self.problem.integrator.n, axis=2)
        sq_density = np.sum(self.problem.integrator.coeffs2weights(self.problem.rots_dcoef) ** 2, axis=0)
        weights *= sq_density[np.newaxis, np.newaxis, :]

        pts_rot = rotated_grids(self.L, self.problem.integrator.rots)

        if self.L % 2 == 0:
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
            / (self.problem.n * self.L ** 4)  # self.problem.n normalization not necessary
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

        return FourierKernel(kernel_f, centered=False)

    def update(self, x0=None, b_coeff=None, tol=None, maxiter=None):
        # TODO implement default b_coeff in case we do have an initial volume
        """ Return an estimate as a Volume instance. """
        if b_coeff is None:
            b_coeff = self.src_backward()

        # if fvol0 is not None:
        #     x0 = fvol0  # self.basis.evaluate_t(vol0.T)  # TODO check whether this is used correctly like this
        # else:
        #     x0 = None
        est_coeff = self.conj_grad(b_coeff, x0=x0, tol=tol, maxiter=maxiter)  # TODO max_it?
        est = self.basis.evaluate(est_coeff)
        self.problem.vol = Volume(est)

        return est_coeff

    def src_backward(self):
        """
        Apply adjoint mapping to source

        :return: The adjoint mapping applied to the images, averaged over the whole dataset and expressed
            as coefficients of `basis`.
        """
        mean_b = np.zeros((self.L, self.L, self.L), dtype=self.dtype)

        batch_mean_b = self.problem.adjoint_forward(self.problem.imgs) / self.n
        mean_b += batch_mean_b.astype(self.dtype)

        res = self.basis.evaluate_t(mean_b.T)  # RCOPT
        logger.info(f"Determined adjoint mappings. Shape = {res.shape}")
        return res

    def conj_grad(self, b_coeff, x0=None, tol=None, maxiter=None):
        n = b_coeff.shape[0]
        kernel = self.kernel

        regularizer = config.mean.regularizer  # TODO fix that we can use regularizer
        if regularizer > 0:
            kernel += regularizer

        operator = LinearOperator(
            (n, n), matvec=partial(self.apply_kernel, kernel=kernel), dtype=self.dtype
        )
        if self.precond_kernel is None:
            M = None
        else:
            precond_kernel = self.precond_kernel
            if regularizer > 0:
                precond_kernel += regularizer
            M = LinearOperator(
                (n, n),
                matvec=partial(self.apply_kernel, kernel=precond_kernel),
                dtype=self.dtype,
            )

        tol = tol or config.mean.cg_tol
        target_residual = tol * norm(b_coeff)

        def cb(xk):
            logger.info(
                f"Delta {norm(b_coeff - self.apply_kernel(xk))} (target {target_residual})"
            )

        x, info = scipy.sparse.linalg.cg(
            operator, b_coeff, M=M, callback=cb, tol=tol, atol=0, x0=x0, maxiter=maxiter
        )

        if info != 0:
            raise RuntimeError("Unable to converge!")
        return x

    def apply_kernel(self, vol_coeff, kernel=None):
        """
        Applies the kernel represented by convolution
        :param vol_coeff: The volume to be convolved, stored in the basis coefficients.
        :param kernel: a Kernel object. If None, the kernel for this Estimator is used.
        :return: The result of evaluating `vol_coeff` in the given basis, convolving with the kernel given by
            kernel, and backprojecting into the basis.
        """
        if kernel is None:
            kernel = self.kernel
        vol = self.basis.evaluate(vol_coeff).T  # RCOPT
        vol = kernel.convolve_volume(vol)
        vol = self.basis.evaluate_t(vol.T)  # RCOPT

        return vol

    def eval_filter_grid(self, power=1):
        grid2d = grid_2d(self.L, dtype=self.dtype)
        omega = np.pi * np.vstack((grid2d["x"].flatten(), grid2d["y"].flatten()))

        filter_values = self.problem.filter.evaluate(omega)
        if power != 1:
            filter_values **= power

        h = np.reshape(filter_values, grid2d["x"].shape)

        return h
