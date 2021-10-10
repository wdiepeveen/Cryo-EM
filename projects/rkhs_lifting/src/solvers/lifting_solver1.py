import numpy as np
import logging

from scipy.fftpack import fft2

from aspire.nufft import anufft
from aspire.reconstruction.kernel import FourierKernel
from aspire.utils.fft import mdim_ifftshift
from aspire.utils.matlab_compat import m_flatten, m_reshape
from aspire.volume import rotated_grids, Volume

from projects.rkhs_lifting.src.plans.lifting_plan1 import Lifting_Plan1
from projects.rkhs_lifting.src.solvers import Joint_Volume_Rots_Solver
from projects.rkhs_lifting.src.solvers.convex_optimisation.ppdhg import Preconditioned_PDHG

logger = logging.getLogger(__name__)


class RKHS_Lifting_Solver1(Joint_Volume_Rots_Solver):
    def __init__(self,
                 vol=None,
                 squared_noise_level=None,
                 density_coeffs=None,
                 dual_coeffs=None,
                 stop=None,  # TODO here a default stopping criterion
                 stop_density_update=None,  # TODO here a default stopping criterion
                 images=None,
                 filter=None,
                 amplitude=None,
                 integrator=None,
                 volume_reg_param=None,
                 rots_density_reg_param=None,
                 dtype=np.float32,
                 seed=0,
                 ):
        plan = Lifting_Plan1(vol=vol,
                             squared_noise_level=squared_noise_level,
                             density_coeffs=density_coeffs,
                             dual_coeffs=dual_coeffs,
                             stop=stop,  # TODO here a default stopping criterion
                             stop_density_update=stop_density_update,  # TODO here a default stopping criterion
                             images=images,
                             filter=filter,
                             amplitude=amplitude,
                             integrator=integrator,
                             volume_reg_param=volume_reg_param,
                             rots_density_reg_param=rots_density_reg_param,
                             dtype=dtype,
                             seed=seed, )

        super().__init__(plan=plan)

    def stop_solver(self):
        # TODO this one should probably go better elsewhere since it is quite default
        return self.iter == self.plan.o.stop  # TODO this now only works since we assume that this is an integer

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
        dtype = self.plan.p.dtype

        # Compute q:
        logger.info("Computing qs")
        im = self.plan.p.images.asnumpy()
        qs = np.zeros((n, N), dtype=self.plan.p.dtype)
        logger.info("Construct qs with batch size {}".format(self.plan.o.rots_batch_size))
        q3 = np.sum(im ** 2, axis=(1, 2))[None, :]
        for start in range(0, self.plan.p.n, self.plan.o.rots_batch_size):
            logger.info(
                "Running through projections {}/{} = {}%".format(start, n, np.round(start / n * 100, 2)))
            rots_sampling_projections = self.plan.forward(self.plan.o.vol, start, self.plan.o.rots_batch_size).asnumpy()

            q1 = np.sum(rots_sampling_projections ** 2, axis=(1, 2))[:, None]
            q2 = - 2 * np.einsum("ijk,gjk->gi", im, rots_sampling_projections)

            all_idx = np.arange(start, min(start + self.plan.o.rots_batch_size, n))
            qs[all_idx, :] = (q1 + q2 + q3) / (2 * self.plan.o.squared_noise_level * L ** 2)

        q = self.plan.p.integrator.coeffs_to_weights(qs)  # TODO this is not correct if we have non-identity integration
        logger.info("Computed qs, shape = {}".format(q.shape))

        A = np.ones((1, n), dtype=dtype)

        # e = np.ones((n, N), dtype=dtype)  # Gives error if we use shape (n,1)
        # A = self.plan.p.integrator.coeffs_to_weights(e)[:, 0].T[None, :]

        # Compute sigmas and taus
        alpha = 1.
        sigmas = 0.9 / np.sum(np.abs(A) ** (2 - alpha), axis=0)  # For the sigmas
        taus = 0.9 / np.sum(np.abs(A) ** alpha, axis=1)  # For the taus

        # Acola = np.sum(np.abs(A) ** (2 - alpha), axis=0)  # For the sigmas
        # Arowa = np.sum(np.abs(A) ** alpha, axis=1)  # For the taus
        # sigmas = np.repeat(1 / Acola[:, None], N, axis=1)
        # taus = np.repeat(1 / Arowa[:, None], N, axis=1)

        def primal_prox(primals, sigma):
            result = 1 / (1 + self.plan.p.rots_density_reg_param * self.plan.p.integrator.kernel.norm ** 2 * sigma) * (
                    primals - sigma / self.plan.p.n * q)
            result *= (result >= 0)
            return result

        def dual_prox(duals, tau):
            result = duals - tau
            return result

        def block_operator(primals):
            return A @ primals

        def adjoint_block_operator(duals):
            return A.T @ duals

        print("Start PPDHG")
        solver = Preconditioned_PDHG(primalProx=primal_prox,
                                     dualProx=dual_prox,
                                     operator=block_operator,
                                     adjoint=adjoint_block_operator,
                                     x0=self.plan.o.density_coeffs,
                                     y0=self.plan.o.dual_coeffs,
                                     sigmas=sigmas[:, None],
                                     taus=taus[:, None])

        solver.solve()
        self.plan.o.density_coeffs = solver.x.astype(dtype)
        self.plan.o.dual_coeffs = solver.y.astype(dtype)

        return solver.normalized_errors

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
