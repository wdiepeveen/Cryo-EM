import numpy as np
import logging

from scipy.fftpack import fft2

from aspire.image import Image
from aspire.nufft import anufft
from aspire.reconstruction.kernel import FourierKernel
from aspire.utils.fft import mdim_ifftshift
from aspire.utils.matlab_compat import m_flatten, m_reshape
from aspire.volume import rotated_grids, Volume

from projects.lifting_v2.src.manifolds.so3 import SO3
from projects.lifting_v2.src.plans.refinement_plan2 import Refinement_Plan2
from projects.lifting_v2.src.solvers import Joint_Volume_Rots_Solver

logger = logging.getLogger(__name__)


class Refinement_Solver2(Joint_Volume_Rots_Solver):
    def __init__(self,
                 quaternions=None,
                 rots_coeffs=None,
                 sigmas=None,
                 tau=None,
                 images=None,
                 filter=None,
                 amplitude=None,
                 dtype=np.float32,
                 seed=0,
                 ):
        plan = Refinement_Plan2(quaternions=quaternions,
                                rots_coeffs=rots_coeffs,
                                sigmas=sigmas,
                                tau=tau,
                                images=images,
                                filter=filter,
                                amplitude=amplitude,
                                rots_batch_size=8192,
                                dtype=dtype,
                                seed=seed,
                                )

        super().__init__(plan=plan)

    def initialize_solver(self):
        # Update data discrepancy
        logger.info("Initialize solver")

    def stop_solver(self):
        return self.iter == 1

    def step_solver(self):
        self.rots_step()

        self.volume_step()

    def finalize_solver(self):
        print("Solver has finished")

    def rots_step(self):
        weights = np.clip(self.plan.rots_coeffs.T, 0.0, 1.0)
        weights /= weights.sum(axis=1)[:, None]

        manifold = SO3()

        quaternions = np.zeros((self.plan.N, 4))

        N_batch_size = 100
        for start in range(0, self.plan.N, N_batch_size):
            N_idx = np.arange(start, min(start + N_batch_size, self.plan.N))
            selected_weights = weights[N_idx]
            # Select columns with rots having non-zero coefficients
            quat_idx = np.arange(0, self.plan.n)[np.sum(selected_weights, axis=0) > 0.]
            # Compute means
            quaternions[N_idx, :] = manifold.mean(self.plan.quaternions[None, None, quat_idx], selected_weights[None, :, quat_idx])[0, 0]
            logger.info("Computing means at {}%".format(int((N_idx[-1] + 1) / self.plan.N * 100)))

        self.plan.quaternions = quaternions

    def volume_step(self):
        L = self.plan.L
        N = self.plan.N
        dtype = self.plan.dtype

        # compute adjoint forward map of images
        logger.info("Compute adjoint forward mapping on the images")
        src = self.plan.adjoint_forward(Image((self.plan.tau / self.plan.sigmas[:, None, None]) * self.plan.images.asnumpy()))

        # compute kernel in fourier domain
        _2L = 2 * L
        kernel = np.zeros((_2L, _2L, _2L), dtype=dtype)
        sq_filters_f = self.plan.eval_filter_grid(power=2)
        sq_filters_f *= self.plan.amplitude ** 2

        for start in range(0, N, self.plan.rots_batch_size):
            logger.info(
                "Computing kernel at {}%".format(np.round(start / N * 100, 2)))
            all_idx = np.arange(start, min(start + self.plan.rots_batch_size, N))
            num_idx = len(all_idx)

            # weights = np.repeat(sq_filters_f[:, :, None], num_idx, axis=2)
            weights = sq_filters_f[:, :, None] * (self.plan.tau / self.plan.sigmas[None, None, :])

            if L % 2 == 0:
                weights[0, :, :] = 0
                weights[:, 0, :] = 0

            weights = m_flatten(weights)

            pts_rot = rotated_grids(L, self.plan.rots[all_idx, :, :])
            # pts_rot = np.moveaxis(pts_rot, 1, 2)  # We don't need this. It was in Aspire. Might be needed for non radial kernels
            pts_rot = m_reshape(pts_rot, (3, -1))

            kernel += (
                    1
                    / (L ** 6)
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
        f_kernel += 1

        f_kernel = FourierKernel(
            1.0 / f_kernel.kernel, centered=False
        )

        # apply kernel
        vol = np.real(f_kernel.convolve_volume(src.T)
                      / (L ** 3)  # Compensation for the lack of scaling in the forward operator
                      ).astype(dtype)

        self.plan.vol = Volume(vol)
