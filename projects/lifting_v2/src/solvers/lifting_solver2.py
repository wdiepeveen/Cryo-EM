import numpy as np
import logging

from scipy.fftpack import fft2, fftn, ifftn

from aspire.nufft import anufft
from aspire.reconstruction.kernel import FourierKernel
from aspire.utils.fft import mdim_ifftshift
from aspire.utils.matlab_compat import m_flatten, m_reshape
from aspire.volume import rotated_grids, Volume

from projects.lifting_v2.src.plans.lifting_plan2 import Lifting_Plan2
from projects.lifting_v2.src.solvers import Joint_Volume_Rots_Solver

logger = logging.getLogger(__name__)


class Lifting_Solver2(Joint_Volume_Rots_Solver):
    def __init__(self,
                 # variables to be optimised
                 vol=None,
                 rots_coeffs=None,
                 squared_noise_level=None,  # sigma
                 volume_reg_param=None,  # tau
                 # data
                 images=None,  # f_i
                 # parameters
                 filter=None,
                 amplitude=None,
                 integrator=None,
                 rots_reg_param=None,  # lambda
                 rots_reg_scaling_param=66 / 100,  # eta
                 # solver options
                 max_iter=None,
                 save_iterates=False,
                 dtype=np.float32,
                 seed=0,
                 ):  # TODO in next solver also add kernel smoothing
        plan = Lifting_Plan2(vol=vol,
                             rots_coeffs=rots_coeffs,
                             squared_noise_level=squared_noise_level,  # sigma
                             volume_reg_param=volume_reg_param,  # tau
                             images=images,
                             filter=filter,
                             amplitude=amplitude,
                             integrator=integrator,
                             rots_reg_param=rots_reg_param,
                             rots_reg_scaling_param=rots_reg_scaling_param,
                             max_iter=max_iter,
                             save_iterates=save_iterates,
                             dtype=dtype,
                             seed=seed)

        super().__init__(plan=plan)

        self.data_discrepancy = np.zeros((self.plan.n, self.plan.N))  # (\|Ag.u - f_i\|^2)_g,i

        print("vol = {}".format(self.plan.vol.asnumpy()))
        print("beta = {}".format(self.plan.rots_coeffs))
        print("sigmas = {}".format(self.plan.sigmas))
        print("tau = {}".format(self.plan.tau))

    def stop_solver(self):
        return self.iter == self.plan.max_iter  # TODO add || relerror/change is small

    def step_solver(self):

        # Compute squared errors so we can use it for both weight update and sigma update

        logger.info("Update data_discrepancies")
        self.data_discrepancy_update()

        logger.info("Do rots update step")
        self.rots_density_step()

        logger.info("Do sigma update step")
        self.sigma_step()
        print("sigmas = {}".format(self.plan.sigmas))

        logger.info("Do vol update step")
        self.volume_step()
        print("volume = {}".format(self.plan.vol.asnumpy()))

        logger.info("Do tau update step")
        self.tau_step()
        print("tau = {}".format(self.plan.tau))

        if self.plan.save_iterates:
            self.vol_iterates.append(self.plan.vol)
            self.rots_coeffs_iterates.append(self.plan.rots_coeffs)

    def finalize_solver(self):
        print("Solver has finished")

    def data_discrepancy_update(self):
        L = self.plan.L
        N = self.plan.N
        n = self.plan.n
        dtype = self.plan.dtype

        logger.info("Computing \|Ag.u - f_i\|^2")
        im = self.plan.images.asnumpy()
        F = np.zeros((n, N), dtype=dtype)
        F3 = np.sum(im ** 2, axis=(1, 2))[None, :]
        print("F3 = {}".format(F3[:, 0]))

        for start in range(0, n, self.plan.rots_batch_size):
            logger.info(
                "Running through projections {}/{} = {}%".format(start, n, np.round(start / n * 100, 2)))
            rots_sampling_projections = self.plan.forward(self.plan.vol, start, self.plan.rots_batch_size).asnumpy()

            F1 = np.sum(rots_sampling_projections ** 2, axis=(1, 2))[:, None]
            print("F1 = {}".format(F1[:, 0]))
            F2 = - 2 * np.einsum("ijk,gjk->gi", im, rots_sampling_projections)
            print("F2 = {}".format(F2[:, 0]))

            all_idx = np.arange(start, min(start + self.plan.rots_batch_size, n))
            F[all_idx, :] = (F1 + F2 + F3) / (L ** 2)  # 2 * self.plan.squared_noise_level missing now

        self.data_discrepancy = F
        print("F = {}".format(F[:, 0]))

    def rots_density_step(self):
        n = self.plan.n
        eta = self.plan.eta
        lambd = self.plan.lambd
        dtype = self.plan.dtype

        self.plan.rots_coeffs = self.projection_simplex(
            - (n ** eta) / lambd * self.data_discrepancy / (2 * self.plan.sigmas[None, :]), axis=0).astype(dtype)

    def sigma_step(self):
        self.plan.sigmas = np.einsum("gi,gi->i", self.data_discrepancy, self.plan.rots_coeffs)

    def tau_step(self):
        self.plan.tau = np.sum(self.plan.vol.asnumpy() ** 2) / (self.plan.L ** 3)

    # def volume_step(  # TODO do everything in Fourier domain to avoid numerical instabilities?
    #         self):  # TODO check whether the sigmas work well like this and whether scaling in tau is okay like this
    #     L = self.plan.L
    #     n = self.plan.n
    #     dtype = self.plan.dtype
    #
    #     # rots_weights = self.plan.integrator.coeffs_to_integrand_weights(self.plan.rots_coeffs)
    #
    #     # compute adjoint forward map of images
    #     logger.info("Compute adjoint forward mapping on the images")
    #     src = self.plan.adjoint_forward(self.plan.images,
    #                                     (self.plan.tau / self.plan.sigmas[None, :]) * self.plan.rots_coeffs)
    #     print("src = {}".format(src))
    #     print("src.shape = {}".format(src.shape))
    #
    #     # compute kernel in fourier domain
    #     _2L = 2 * L
    #     kernel = np.zeros((_2L, _2L, _2L), dtype=dtype)
    #     sq_filters_f = self.plan.eval_filter_grid(power=2)
    #     sq_filters_f *= self.plan.amplitude ** 2
    #
    #     summed_rots_weights = np.sum((self.plan.tau / self.plan.sigmas[None, :]) * self.plan.rots_coeffs, axis=1)
    #
    #     for start in range(0, self.plan.n, self.plan.rots_batch_size):
    #         logger.info(
    #             "Running through projections {}/{} = {}%".format(start, n, np.round(start / n * 100, 2)))
    #         all_idx = np.arange(start, min(start + self.plan.rots_batch_size, n))
    #
    #         # print("summed densities = {}".format(summed_density))
    #         summed_rots_weights_ = summed_rots_weights[all_idx]
    #         weights = sq_filters_f[:, :, None] * summed_rots_weights_[None, None, :]
    #
    #         if L % 2 == 0:
    #             weights[0, :, :] = 0
    #             weights[:, 0, :] = 0
    #
    #         weights = m_flatten(weights)
    #
    #         pts_rot = rotated_grids(L, self.plan.integrator.rots[all_idx, :, :])
    #         pts_rot = np.moveaxis(pts_rot, 1, 2)  # TODO do we need this? -> No, but was in Aspire. Might be needed for non radial kernels
    #         pts_rot = m_reshape(pts_rot, (3, -1))
    #
    #         kernel += (
    #                 1
    #                 / (L ** 4)  # TODO check whether scaling is correct like this
    #                 * anufft(weights, pts_rot, (_2L, _2L, _2L), real=True)
    #         )
    #
    #     print("kernel = {}".format(kernel))
    #
    #     # Ensure symmetric kernel
    #     kernel[0, :, :] = 0
    #     kernel[:, 0, :] = 0
    #     kernel[:, :, 0] = 0
    #
    #     logger.info("Computing non-centered Fourier Transform")
    #     kernel = mdim_ifftshift(kernel, range(0, 3))
    #     # kernel_f = fft2(kernel, axes=(0, 1, 2))
    #     kernel_f = fftn(kernel, axes=(0, 1, 2))
    #     kernel_f = np.real(kernel_f)
    #     kernel_f += 1.  # TODO check whether this works
    #
    #     f_kernel = FourierKernel(kernel_f, centered=False)
    #     # f_kernel += 1.
    #
    #     # f_kernel = FourierKernel(
    #     #     1.0 / f_kernel.kernel, centered=False
    #     # )
    #
    #     # apply kernel
    #     src_f = fftn(src, (_2L, _2L, _2L))
    #     vol = np.real(ifftn(src_f/kernel_f)[:L,:L,:L]).astype(dtype)
    #     # vol = np.real(f_kernel.convolve_volume(src.T)).astype(
    #     #     dtype)  # TODO works, but still not entirely sure why we need to transpose here
    #
    #     self.plan.vol = Volume(vol)

    def volume_step(  # OLD version
            self):  # TODO check whether the sigmas work well like this and whether scaling in tau is okay like this
        L = self.plan.L
        n = self.plan.n
        dtype = self.plan.dtype

        # rots_weights = self.plan.integrator.coeffs_to_integrand_weights(self.plan.rots_coeffs)

        # compute adjoint forward map of images
        logger.info("Compute adjoint forward mapping on the images")
        src = self.plan.adjoint_forward(self.plan.images,
                                        (self.plan.tau / self.plan.sigmas[None, :]) * self.plan.rots_coeffs)
        print("src = {}".format(src))

        # compute kernel in fourier domain
        _2L = 2 * L
        kernel = np.zeros((_2L, _2L, _2L), dtype=dtype)
        sq_filters_f = self.plan.eval_filter_grid(power=2)
        sq_filters_f *= self.plan.amplitude ** 2

        summed_rots_weights = np.sum((self.plan.tau / self.plan.sigmas[None, :]) * self.plan.rots_coeffs, axis=1)

        for start in range(0, self.plan.n, self.plan.rots_batch_size):
            logger.info(
                "Running through projections {}/{} = {}%".format(start, n, np.round(start / n * 100, 2)))
            all_idx = np.arange(start, min(start + self.plan.rots_batch_size, n))

            # print("summed densities = {}".format(summed_density))
            summed_rots_weights_ = summed_rots_weights[all_idx]
            weights = sq_filters_f[:, :, None] * summed_rots_weights_[None, None, :]

            if L % 2 == 0:
                weights[0, :, :] = 0
                weights[:, 0, :] = 0

            weights = m_flatten(weights)

            pts_rot = rotated_grids(L, self.plan.integrator.rots[all_idx, :, :])
            # pts_rot = np.moveaxis(pts_rot, 1, 2)  # TODO do we need this? -> No, but was in Aspire. Might be needed for non radial kernels
            pts_rot = m_reshape(pts_rot, (3, -1))

            kernel += (
                    1
                    / (L ** 4)  # TODO check whether scaling is correct like this
                    * anufft(weights, pts_rot, (_2L, _2L, _2L), real=True)
            )

        print("kernel = {}".format(kernel))

        # Ensure symmetric kernel
        kernel[0, :, :] = 0
        kernel[:, 0, :] = 0
        kernel[:, :, 0] = 0

        logger.info("Computing non-centered Fourier Transform")
        kernel = mdim_ifftshift(kernel, range(0, 3))
        kernel_f = fft2(kernel, axes=(0, 1, 2))
        kernel_f = np.real(kernel_f)
        # kernel_f += 1.  # TODO check whether this works

        f_kernel = FourierKernel(kernel_f, centered=False)
        f_kernel += 1.

        f_kernel = FourierKernel(
            1.0 / f_kernel.kernel, centered=False
        )

        # apply kernel
        vol = np.real(f_kernel.convolve_volume(src.T)).astype(
            dtype)  # TODO works, but still not entirely sure why we need to transpose here

        self.plan.vol = Volume(vol)

    def projection_simplex(self, V, z=1, axis=None):
        """
        Projection of x onto the simplex, scaled by z:
            P(x; z) = argmin_{y >= 0, sum(y) = z} ||y - x||^2
        z: float or array
            If array, len(z) must be compatible with V
        axis: None or int
            axis=None: project V by P(V.ravel(); z)
            axis=1: project each V[i] by P(V[i]; z[i])
            axis=0: project each V[:, j] by P(V[:, j]; z[j])
        # Function from: https://gist.github.com/mblondel/c99e575a5207c76a99d714e8c6e08e89
        # Author: Mathieu Blondel
        """
        if axis == 1:
            print("len(V) = {}".format(len(V)))
            n_features = V.shape[1]
            U = np.sort(V, axis=1)[:, ::-1]
            print("U = {}".format(U[0]))
            z = np.ones(len(V)) * z
            cssv = np.cumsum(U, axis=1) - z[:, np.newaxis]
            print("cssv = {}".format(cssv[1]))
            ind = np.arange(n_features) + 1
            cond = U - cssv / ind > 0
            rho = np.count_nonzero(cond, axis=1)
            theta = cssv[np.arange(len(V)), rho - 1] / rho
            return np.maximum(V - theta[:, np.newaxis], 0)

        elif axis == 0:
            return self.projection_simplex(V.T, z, axis=1).T

        else:
            V = V.ravel().reshape(1, -1)
            return self.projection_simplex(V, z, axis=1).ravel()
