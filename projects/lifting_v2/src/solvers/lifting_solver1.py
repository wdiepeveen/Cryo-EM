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
                 rots_coeffs_reg_param=None,  # \lambda2
                 rots_coeffs_reg_scaling_param=66/100,  # p
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
                             rots_coeffs_reg_scaling_param=rots_coeffs_reg_scaling_param,
                             dtype=dtype,
                             seed=seed, )

        super().__init__(plan=plan)

    def stop_solver(self):
        return self.iter == self.plan.o.max_iter  # TODO add || relerror/change is small

    def step_solver(self):
        self.rots_density_step()
        # self.cost.append(self.plan.get_cost())

        self.volume_step()
        # self.cost.append(self.plan.get_cost())

    def finalize_solver(self):
        print("Solver has finished")

    def rots_density_step(self):
        L = self.plan.p.L
        N = self.plan.p.N
        n = self.plan.p.n
        p = self.plan.p.rots_coeffs_reg_scaling_param
        lamb = self.plan.p.rots_coeffs_reg_param
        dtype = self.plan.p.dtype

        # Compute q:
        logger.info("Computing F_i(g)")
        im = self.plan.p.images.asnumpy()
        F = np.zeros((n, N), dtype=self.plan.p.dtype)
        logger.info("Construct F_i(g) with g batch size {}".format(self.plan.o.rots_batch_size))
        F3 = np.sum(im ** 2, axis=(1, 2))[None, :]

        for start in range(0, self.plan.p.n, self.plan.o.rots_batch_size):
            logger.info(
                "Running through projections {}/{} = {}%".format(start, n, np.round(start / n * 100, 2)))
            rots_sampling_projections = self.plan.forward(self.plan.o.vol, start, self.plan.o.rots_batch_size).asnumpy()

            F1 = np.sum(rots_sampling_projections ** 2, axis=(1, 2))[:, None]
            F2 = - 2 * np.einsum("ijk,gjk->gi", im, rots_sampling_projections)

            all_idx = np.arange(start, min(start + self.plan.o.rots_batch_size, n))
            F[all_idx, :] = (F1 + F2 + F3) / (2 * self.plan.o.squared_noise_level * L ** 2)

        def projection_simplex(V, z=1, axis=None):
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
                return projection_simplex(V.T, z, axis=1).T

            else:
                V = V.ravel().reshape(1, -1)
                return projection_simplex(V, z, axis=1).ravel()

        self.plan.o.rots_coeffs = projection_simplex(-n**p/lamb * F, axis=0).astype(dtype)

    def volume_step(self):
        L = self.plan.p.L
        n = self.plan.p.n
        dtype = self.plan.p.dtype

        evaluated_density = self.plan.p.integrator.coeffs_to_weights(self.plan.o.density_coeffs)

        # compute adjoint forward map of images
        logger.info("Compute adjoint forward mapping on the images")
        src = self.plan.adjoint_forward(self.plan.p.images, evaluated_density)

        # compute kernel in fourier domain

        _2L = 2 * L
        kernel = np.zeros((_2L, _2L, _2L), dtype=dtype)
        sq_filters_f = self.plan.eval_filter_grid(power=2)
        sq_filters_f *= self.plan.p.amplitude ** 2

        summed_density = np.sum(evaluated_density, axis=1)  # TODO check whether axis=1 instead of 0 is correct

        for start in range(0, self.plan.p.n, self.plan.o.rots_batch_size):
            logger.info(
                "Running through projections {}/{} = {}%".format(start, n, np.round(start / n * 100, 2)))
            all_idx = np.arange(start, min(start + self.plan.o.rots_batch_size, n))

            # weights = np.repeat(sq_filters_f[:, :, None], self.plan.o.rots_batch_size, axis=2)

            # print("summed densities = {}".format(summed_density))
            summed_density_ = summed_density[all_idx]
            weights = sq_filters_f[:, :, None] * summed_density_[None, None, :]  # [np.newaxis, np.newaxis, :]

            if L % 2 == 0:
                weights[0, :, :] = 0
                weights[:, 0, :] = 0

            weights = m_flatten(weights)

            pts_rot = rotated_grids(L, self.plan.p.integrator.rots[all_idx, :, :])
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
        f_kernel += self.plan.p.volume_reg_param * self.plan.o.squared_noise_level

        f_kernel = FourierKernel(
            1.0 / f_kernel.kernel, centered=False
        )

        # apply kernel
        vol = np.real(f_kernel.convolve_volume(src.T)).astype(
            dtype)  # TODO works, but still not entirely sure why we need to transpose here

        self.plan.o.vol = Volume(vol)
