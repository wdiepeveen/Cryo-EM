import numpy as np
import logging

from scipy.fftpack import fft2, fftn, ifftn

from aspire.image import Image
from aspire.nufft import anufft
from aspire.reconstruction.kernel import FourierKernel
from aspire.utils.fft import mdim_ifftshift
from aspire.utils.matlab_compat import m_flatten, m_reshape
from aspire.volume import rotated_grids, Volume

from projects.lifting_v2.src.manifolds.so3 import SO3
from projects.lifting_v2.src.plans.lifting_plan3 import Lifting_Plan3
from projects.lifting_v2.src.solvers import Joint_Volume_Rots_Solver

logger = logging.getLogger(__name__)


class Lifting_Solver3(Joint_Volume_Rots_Solver):
    def __init__(self,
                 # variables to be optimised
                 vol=None,
                 # data
                 images=None,  # f_i
                 # parameters
                 pixel_size=5,
                 filter=None,
                 amplitude=None,
                 squared_noise_level=None,  # sigma
                 volume_reg_param=None,  # tau
                 volume_kernel_reg_param=None,  # tau2
                 integrator=None,
                 rots_reg_scaling_param=66 / 100,  # eta
                 J0=None,
                 rots_reg_param_range=None,
                 # solver options
                 max_iter=None,
                 save_iterates=False,
                 dtype=np.float32,
                 seed=0,
                 debug=False,
                 # experiment=None,
                 ):
        plan = Lifting_Plan3(vol=vol,
                             squared_noise_level=squared_noise_level,  # sigma
                             volume_reg_param=volume_reg_param,  # tau1
                             volume_kernel_reg_param=volume_kernel_reg_param,  # tau2
                             images=images,
                             pixel_size=pixel_size,
                             filter=filter,
                             amplitude=amplitude,
                             integrator=integrator,
                             # rots_reg_param=rots_reg_param,
                             rots_reg_scaling_param=rots_reg_scaling_param,
                             J0=J0,
                             rots_reg_param_range=rots_reg_param_range,
                             max_iter=max_iter,
                             save_iterates=save_iterates,
                             dtype=dtype,
                             seed=seed)

        super().__init__(plan=plan)

        self.debug = debug  # TODO setup debug routine that asserts that the cost goes down every part of the iteration
        # self.experiment = experiment

        self.quaternions_iterates = []
        self.rots_iterates = []

        # print("vol = {}".format(self.plan.vol.asnumpy()))
        # print("beta = {}".format(self.plan.rots_coeffs))
        # print("sigmas = {}".format(self.plan.sigmas))
        # print("tau = {}".format(self.plan.tau))

    def initialize_solver(self):
        logger.info("Initialising Solver")
        # self.cost.append(self.plan.get_cost())

    def stop_solver(self):
        return self.iter == self.plan.max_iter

    def step_solver(self):
        # Update data discrepancy
        logger.info("Update data_discrepancies")
        self.plan.data_discrepancy_update()

        # Compute squared errors so we can use it for both weight update and sigma update
        # if self.experiment is None:
        if self.plan.J is not None:
            logger.info("Do lambda update step")
            self.lambda_step()
        print("lambda = {}".format(self.plan.lambd))

        logger.info("Do rots update step")
        self.rots_step()
        self.cost.append(self.plan.get_cost())
        # print("betas = {}".format(self.plan.rots_coeffs))

        logger.info("Do vol update step")
        self.volume_step()
        self.cost.append(self.plan.get_cost())
        # print("volume = {}".format(self.plan.vol.asnumpy()))

        # elif self.experiment == "consistency":
        #     logger.info("Do rots update step")
        #     self.rots_step()
        #     self.cost.append(self.plan.get_cost())
        #     # print("betas = {}".format(self.plan.rots_coeffs))

        if self.plan.save_iterates:
            self.vol_iterates.append(self.plan.vol)
            self.rots_coeffs_iterates.append(self.plan.rots_coeffs)
            self.quaternions_iterates.append(self.plan.quaternions)
            self.rots_iterates.append(self.plan.rots)
            self.sigmas_iterates.append(self.plan.sigmas)
            self.tau_iterates.append(self.plan.tau)

    def finalize_solver(self):
        print("Solver has finished")

    def lambda_step(self):
        F = self.plan.data_discrepancy / (2 * self.plan.sigmas[None, :])
        Fj = np.sort(F, axis=0)
        FJ = Fj[0:self.plan.J]
        summed_FJ = np.sum(FJ, axis=0)
        lambdas = 1/2 * self.plan.J0 ** (5/3) * (self.plan.n/self.plan.J) ** (2/3) * (Fj[self.plan.J] - 1 / self.plan.J * summed_FJ)
        # lambdas = self.plan.J * (self.plan.n ** self.plan.eta) * (Fj[self.plan.J] - 1 / self.plan.J * summed_FJ)
        self.plan.lambd = lambdas + 1e-16

    def rots_step(self):
        # Stage 1: compute weights
        n = self.plan.n
        N = self.plan.N
        eta = self.plan.eta
        dtype = self.plan.dtype

        # self.plan.rots_coeffs = self.projection_simplex(
        #     - (n ** eta) / self.plan.lambd[None, :] * self.plan.data_discrepancy / (2 * self.plan.sigmas[None, :]),
        #     axis=0).astype(dtype)

        rots_coeffs = np.zeros((n, N), dtype=dtype)
        for start in range(0, N, self.plan.img_batch_size):
            all_idx = np.arange(start, min(start + self.plan.img_batch_size, N))

            rots_coeffs[:, all_idx] = self.projection_simplex(
                - (n ** eta) / self.plan.lambd[None, all_idx] * self.plan.data_discrepancy[:, all_idx] / (
                        2 * self.plan.sigmas[None, all_idx]), axis=0).astype(dtype)

            logger.info(
                "Projecting {} vectors onto {}-simplex at {}%".format(N, n, int((all_idx[-1] + 1) / N * 100)))

        self.plan.rots_coeffs = rots_coeffs

        # Stage 2: project measure onto SO(3)
        weights = np.clip(self.plan.rots_coeffs.T, 0.0, 1.0)
        weights /= weights.sum(axis=1)[:, None]

        manifold = SO3()
        quaternions = np.zeros((N, 4))
        for start in range(0, N, self.plan.img_batch_size):
            N_idx = np.arange(start, min(start + self.plan.img_batch_size, N))
            selected_weights = weights[N_idx]
            # Select columns with rots having non-zero coefficients
            quat_idx = np.arange(0, n)[np.sum(selected_weights, axis=0) > 0.]
            # Compute means
            quaternions[N_idx, :] = manifold.mean(self.plan.integrator.quaternions[None, None, quat_idx],
                                                  selected_weights[None, :, quat_idx])[0, 0]
            logger.info("Computing {} means at {}%".format(N, int((N_idx[-1] + 1) / N * 100)))

        self.plan.quaternions = quaternions

    def volume_step(self):
        L = self.plan.L
        N = self.plan.N
        dtype = self.plan.dtype

        # compute adjoint forward map of images
        logger.info("Compute adjoint forward mapping on the images")
        imgs = Image((self.plan.tau / self.plan.sigmas[:, None, None]) * self.plan.images.asnumpy())
        src = np.zeros((L, L, L), dtype=self.plan.dtype)
        for start in range(0, N, self.plan.rots_batch_size):
            all_idx = np.arange(start, min(start + self.plan.rots_batch_size, N))
            src += self.plan.adjoint_forward(Image(imgs[all_idx]), self.plan.rots[all_idx])
            logger.info(
                "Computing adjoint forward mappings from {} rotations at {}%".format(N, int((all_idx[-1] + 1) / N * 100)))

        # compute kernel in fourier domain
        _2L = 2 * L
        kernel = np.zeros((_2L, _2L, _2L), dtype=dtype)
        sq_filters_f = self.plan.eval_filter_grid(power=2)
        sq_filters_f *= self.plan.amplitude ** 2

        for start in range(0, N, self.plan.rots_batch_size):
            all_idx = np.arange(start, min(start + self.plan.rots_batch_size, N))

            # weights = np.repeat(sq_filters_f[:, :, None], num_idx, axis=2)
            weights = sq_filters_f[:, :, None] * (self.plan.tau / self.plan.sigmas[None, None, all_idx])
            # weights = sq_filters_f[:, :, None] * (self.plan.tau / self.plan.sigmas[None, None, :])

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

            logger.info(
                "Computing kernel from {} rotations at {}%".format(N, int((all_idx[-1] + 1) / N * 100)))

        # Ensure symmetric kernel
        kernel[0, :, :] = 0
        kernel[:, 0, :] = 0
        kernel[:, :, 0] = 0

        logger.info("Computing non-centered Fourier Transform")
        kernel = mdim_ifftshift(kernel, range(0, 3))
        kernel_f = fft2(kernel, axes=(0, 1, 2))
        kernel_f = np.real(kernel_f)

        if self.plan.tau2 is not None:
            logger.info("Add ramp filter regularisation")
            # kernel_f += 1 / (L ** 6) * self.plan.tau / self.plan.tau2 * self.plan.vol_reg_kernel ** 20  # TODO get rid of 1/L6
            kernel_f += self.plan.tau / self.plan.tau2 * (self.plan.vol_reg_kernel ** 2)
            # kernel_f += 1 / (L ** 4) * self.plan.tau / self.plan.tau2 * (self.plan.vol_reg_kernel ** 2)
            # TODO check what happens if we have the zero frequency in the top corner (so also no shift in constructor)

        f_kernel = FourierKernel(kernel_f, centered=False)
        f_kernel += 1

        f_kernel = FourierKernel(
            1.0 / f_kernel.kernel, centered=False
        )

        # apply kernel
        vol = np.real(f_kernel.convolve_volume(src.T)
                      # / (L ** 2)  # Compensation for the lack of scaling in the forward operator (according to ASPIRE)
                      # / (L ** 3)  # Compensation for the lack of scaling in the forward operator
                      ).astype(dtype)

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
            # print("len(V) = {}".format(len(V)))
            n_features = V.shape[1]
            U = np.sort(V, axis=1)[:, ::-1]
            # print("U = {}".format(U[0]))
            z = np.ones(len(V)) * z
            cssv = np.cumsum(U, axis=1) - z[:, np.newaxis]
            # print("cssv = {}".format(cssv[1]))
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
