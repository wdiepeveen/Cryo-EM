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
        self.volume_step()
        self.cost.append(self.plan.get_cost())

    def finalize_solver(self):
        print("Solver has finished")

    def rots_density_step(self):
        L = self.plan.p.L
        N = self.plan.p.N
        n = self.plan.p.n
        dtype = self.plan.p.dtype

        # Compute q:
        logger.info("Computing qs")
        rots_sampling_projections = self.plan.forward(self.plan.o.vol).asnumpy()
        im = self.plan.p.images.asnumpy()

        q1 = np.repeat(np.sum(rots_sampling_projections ** 2, axis=(1, 2))[:, None], N, axis=1)
        q2 = - 2 * np.einsum("ijk,gjk->gi", im, rots_sampling_projections)
        q3 = np.repeat(np.sum(im ** 2, axis=(1, 2))[None, :], n, axis=0)
        qs = (q1 + q2 + q3) / (2 * self.plan.o.squared_noise_level * L ** 2)

        q = self.plan.p.integrator.kernel.matrix_mult(qs)
        logger.info("Computed qs")

        A = self.plan.p.integrator.coeffs_to_weights(np.ones((n,), dtype=dtype))[None, :]

        # Compute sigmas and taus
        alpha = 1.
        sigmas = 1 / np.sum(np.abs(A) ** (2 - alpha), axis=0)  # For the sigmas
        taus = 1 / np.sum(np.abs(A) ** alpha, axis=1)  # For the taus

        # Acola = np.sum(np.abs(A) ** (2 - alpha), axis=0)  # For the sigmas
        # Arowa = np.sum(np.abs(A) ** alpha, axis=1)  # For the taus
        # sigmas = np.repeat(1 / Acola[:, None], N, axis=1)
        # taus = np.repeat(1 / Arowa[:, None], N, axis=1)

        def primal_prox(primals, sigma):
            result = 1 / (1 + self.plan.p.rots_density_reg_param * self.plan.p.n * sigma[:, None]) * (
                    primals - sigma[:, None] / self.plan.p.n * q)
            result *= (result >= 0)
            return result

        def dual_prox(duals, tau):
            result = duals - tau[:, None]
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
                                     x0=self.plan.o.rots_dcoef,
                                     y0=self.plan.o.dual_coeffs,
                                     sigmas=sigmas,
                                     taus=taus)

        solver.solve()
        self.plan.o.rots_dcoef = solver.x
        self.plan.o.dual_coeffs = solver.y

        return solver.normalized_errors

    def volume_step(self):
        L = self.plan.p.L
        n = self.plan.p.n
        dtype = self.plan.p.dtype

        # compute adjoint forward map of images
        src = self.plan.adjoint_forward(self.plan.p.images)

        # compute kernel in fourier domain

        _2L = 2 * L
        kernel = np.zeros((_2L, _2L, _2L), dtype=dtype)
        sq_filters_f = self.plan.eval_filter_grid(power=2)
        sq_filters_f *= self.plan.p.amplitude ** 2

        weights = np.repeat(sq_filters_f[:, :, np.newaxis], n, axis=2)

        summed_density = np.sum(self.plan.p.integrator.coeffs_to_weights(self.plan.o.rots_dcoef),
                                axis=1)  # TODO check whether axis=1 instead of 0 is correct
        # print("summed densities = {}".format(summed_density))
        weights *= summed_density  # [np.newaxis, np.newaxis, :]

        pts_rot = rotated_grids(L, self.plan.p.integrator.rots)
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
        f_kernel += self.plan.p.volume_reg_param * self.plan.o.squared_noise_level

        f_kernel = FourierKernel(
            1.0 / f_kernel.kernel, centered=False
        )

        # apply kernel
        vol = np.real(f_kernel.convolve_volume(src.T)).astype(
            dtype)  # TODO works, but still not entirely sure why we need to transpose here

        self.plan.o.vol = Volume(vol)
