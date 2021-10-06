import numpy as np
import logging

from scipy.fftpack import fft2

from aspire.nufft import anufft
from aspire.reconstruction.kernel import FourierKernel
from aspire.utils.fft import mdim_ifftshift
from aspire.utils.matlab_compat import m_flatten, m_reshape
from aspire.volume import rotated_grids, Volume

from projects.rkhs_lifting.src.plans.lifting_plan2 import Lifting_Plan2
from projects.rkhs_lifting.src.solvers import Joint_Volume_Rots_Solver
from projects.rkhs_lifting.src.solvers.convex_optimisation.drs import Douglas_Rachford_Splitting

logger = logging.getLogger(__name__)


class RKHS_Lifting_Solver2(Joint_Volume_Rots_Solver):
    def __init__(self,
                 vol=None,
                 squared_noise_level=None,
                 density_coeffs=None,
                 drs_coeffs=None,
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
        plan = Lifting_Plan2(vol=vol,
                             squared_noise_level=squared_noise_level,
                             density_coeffs=density_coeffs,
                             drs_coeffs=drs_coeffs,
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
        logger.info("Construct qs with batch size {}".format(self.plan.o.batch_size))
        q3 = np.sum(im ** 2, axis=(1, 2))[None, :]
        for start in range(0, self.plan.p.n, self.plan.o.batch_size):
            logger.info(
                "Running through projections {}/{} = {}%".format(start, n, np.round(start / n * 100, 2)))
            rots_sampling_projections = self.plan.forward(self.plan.o.vol, start, self.plan.o.batch_size).asnumpy()

            q1 = np.sum(rots_sampling_projections ** 2, axis=(1, 2))[:, None]
            q2 = - 2 * np.einsum("ijk,gjk->gi", im, rots_sampling_projections)

            all_idx = np.arange(start, min(start + self.plan.o.batch_size, n))
            qs[all_idx, :] = (q1 + q2 + q3) / (2 * self.plan.o.squared_noise_level * L ** 2)

        q = self.plan.p.integrator.coeffs_to_weights(qs)  # TODO this is not correct if we have non-identity integration
        logger.info("Computed qs, shape = {}".format(q.shape))


        def proxf(y):
            result = 1 / (1 + self.plan.p.rots_density_reg_param * n ) * (
                    y - 1 / n * q)
            result *= (result >= 0)
            return result

        def proxg(y):
            proj = np.ones((1, n), dtype=dtype) @ y
            print("proj.shape = {}".format(proj.shape))
            result = y - (1 - proj)/n * y
            return result

        print("Start DRS")
        solver = Douglas_Rachford_Splitting(proxf=proxf,
                                     proxg=proxg,
                                     x0=self.plan.o.drs_coeffs,
                                            )

        solver.solve()
        self.plan.o.drs_coeffs = solver.x.astype(dtype)
        self.plan.o.density_coeffs = proxf(self.plan.o.drs_coeffs).astype(dtype)

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

        summed_density = np.sum(self.plan.p.integrator.coeffs_to_weights(self.plan.o.density_coeffs),
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
