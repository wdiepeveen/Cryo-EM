import numpy as np
import logging

from scipy.fftpack import fft2

from aspire.nufft import anufft
from aspire.reconstruction.kernel import FourierKernel
from aspire.utils.fft import mdim_ifftshift
from aspire.utils.matlab_compat import m_flatten, m_reshape
from aspire.volume import rotated_grids, Volume

from projects.lifting_v2.src.plans.lifting_plan1 import Lifting_Plan1
from projects.lifting_v2.src.solvers import Joint_Volume_Rots_Solver

logger = logging.getLogger(__name__)


class Lifting_Solver1(Joint_Volume_Rots_Solver):
    def __init__(self,
                 vol=None,
                 rots_coeffs=None,
                 max_iter=None,
                 squared_noise_level=None,
                 images=None,
                 filter=None,
                 amplitude=None,
                 integrator=None,
                 volume_reg_param=None,  # \lambda1
                 rots_coeffs_reg_param=None,  # \lambda2 or (\lambda2_init, \lambda2_inf)
                 rots_coeffs_reg_param_rate=1,
                 rots_coeffs_reg_scaling_param=66 / 100,  # p
                 save_iterates=False,
                 dtype=np.float32,
                 seed=0,
                 ):  # TODO in next solver also add kernel smoothing
        plan = Lifting_Plan1(vol=vol,
                             rots_coeffs=rots_coeffs,
                             max_iter=max_iter,
                             squared_noise_level=squared_noise_level,
                             images=images,
                             filter=filter,
                             amplitude=amplitude,
                             integrator=integrator,
                             volume_reg_param=volume_reg_param,
                             rots_reg_param=rots_coeffs_reg_param,
                             rots_coeffs_reg_param_rate=rots_coeffs_reg_param_rate,
                             rots_coeffs_reg_scaling_param=rots_coeffs_reg_scaling_param,
                             save_iterates=save_iterates,
                             dtype=dtype,
                             seed=seed)

        super().__init__(plan=plan)

    def stop_solver(self):
        return self.iter == self.plan.max_iter  # TODO add || relerror/change is small

    def step_solver(self):
        logger.info("Update regularisation parameter lam2")
        self.plan.lam2 = self.plan.lam2_init * np.exp(self.plan.lam2_rate * (1 - self.iter)) + \
                         self.plan.lam2_inf * (1 - np.exp(self.plan.lam2_rate * (1 - self.iter)))

        print(self.plan.lam2)

        logger.info("Do rots update step")
        self.rots_density_step()
        # self.cost.append(self.plan.get_cost())

        logger.info("Do vol update step")
        self.volume_step()
        # self.cost.append(self.plan.get_cost())

        if self.plan.save_iterates:
            self.vol_iterates.append(self.plan.vol)
            self.rots_coeffs_iterates.append(self.plan.rots_coeffs)

    def finalize_solver(self):
        print("Solver has finished")

    def rots_density_step(self):
        L = self.plan.L
        N = self.plan.N
        n = self.plan.n
        p = self.plan.p
        lam = self.plan.lam2
        dtype = self.plan.dtype

        # Compute q:
        logger.info("Computing F_i(g)")
        im = self.plan.images.asnumpy()
        F = np.zeros((n, N), dtype=self.plan.dtype)
        logger.info("Construct F_i(g) with g batch size {}".format(self.plan.rots_batch_size))
        F3 = np.sum(im ** 2, axis=(1, 2))[None, :]

        for start in range(0, n, self.plan.rots_batch_size):
            logger.info(
                "Running through projections {}/{} = {}%".format(start, n, np.round(start / n * 100, 2)))
            rots_sampling_projections = self.plan.forward(self.plan.vol, start, self.plan.rots_batch_size).asnumpy()

            F1 = np.sum(rots_sampling_projections ** 2, axis=(1, 2))[:, None]
            F2 = - 2 * np.einsum("ijk,gjk->gi", im, rots_sampling_projections)

            all_idx = np.arange(start, min(start + self.plan.rots_batch_size, n))
            F[all_idx, :] = (F1 + F2 + F3) / (2 * self.plan.squared_noise_level * L ** 2)

        FF = self.plan.integrator.integrands_to_coeff_weights(F)  # TODO test

        self.plan.rots_coeffs = self.projection_simplex(- (n ** p) / lam * FF, axis=0).astype(dtype)

    def volume_step(self):
        L = self.plan.L
        nn = self.plan.nn
        dtype = self.plan.dtype

        rots_weights = self.plan.integrator.coeffs_to_integrand_weights(self.plan.rots_coeffs)

        # compute adjoint forward map of images
        logger.info("Compute adjoint forward mapping on the images")
        src = self.plan.adjoint_forward(self.plan.images, rots_weights)

        # compute kernel in fourier domain

        _2L = 2 * L
        kernel = np.zeros((_2L, _2L, _2L), dtype=dtype)
        sq_filters_f = self.plan.eval_filter_grid(power=2)
        sq_filters_f *= self.plan.amplitude ** 2

        summed_rots_weights = np.sum(rots_weights, axis=1)  # TODO check whether axis=1 instead of 0 is correct

        for start in range(0, self.plan.nn, self.plan.rots_batch_size):
            logger.info(
                "Running through projections {}/{} = {}%".format(start, nn, np.round(start / nn * 100, 2)))
            all_idx = np.arange(start, min(start + self.plan.rots_batch_size, nn))

            # print("summed densities = {}".format(summed_density))
            summed_rots_weights_ = summed_rots_weights[all_idx]
            weights = sq_filters_f[:, :, None] * summed_rots_weights_[None, None, :]

            if L % 2 == 0:
                weights[0, :, :] = 0
                weights[:, 0, :] = 0

            weights = m_flatten(weights)

            pts_rot = rotated_grids(L, self.plan.integrator.rotsrots[all_idx, :, :])
            # pts_rot = np.moveaxis(pts_rot, 1, 2)  # TODO do we need this? -> No, but was in Aspire. Might be needed for non radial kernels
            pts_rot = m_reshape(pts_rot, (3, -1))

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
        f_kernel += self.plan.lam1 * self.plan.squared_noise_level

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
            n_features = V.shape[1]
            U = np.sort(V, axis=1)[:, ::-1]
            z = np.ones(len(V)) * z
            cssv = np.cumsum(U, axis=1) - z[:, np.newaxis]
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
