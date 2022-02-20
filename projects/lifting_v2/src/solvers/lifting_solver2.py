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
                 debug=False,
                 experiment=None,
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

        self.debug = debug  # TODO setup debug routine that asserts that the cost goes down every part of the iteration
        self.experiment = experiment

        # print("vol = {}".format(self.plan.vol.asnumpy()))
        # print("beta = {}".format(self.plan.rots_coeffs))
        # print("sigmas = {}".format(self.plan.sigmas))
        # print("tau = {}".format(self.plan.tau))

    def initialize_solver(self):
        # Update data discrepancy
        logger.info("Update data_discrepancies")
        self.plan.data_discrepancy_update()
        self.cost.append(self.plan.get_cost())

    def stop_solver(self):
        return self.iter == self.plan.max_iter  # TODO add || relerror/change is small

    def step_solver(self):

        # Compute squared errors so we can use it for both weight update and sigma update
        if self.experiment is None:
            logger.info("Do rots update step")
            self.rots_density_step()
            self.cost.append(self.plan.get_cost())
            # print("betas = {}".format(self.plan.rots_coeffs))

            logger.info("Do vol update step")
            self.volume_step()
            self.cost.append(self.plan.get_cost())
            # print("volume = {}".format(self.plan.vol.asnumpy()))

            logger.info("Do sigma update step")
            self.sigma_step()
            self.cost.append(self.plan.get_cost())
            # print("sigmas = {}".format(self.plan.sigmas))

            # logger.info("Do tau update step")  # TODO 19-02-22 -> skip this
            # self.tau_step()
            # self.cost.append(self.plan.get_cost())
            # print("tau = {}".format(self.plan.tau))

        elif self.experiment == "consistency":
            logger.info("Do rots update step")
            self.rots_density_step()
            self.cost.append(self.plan.get_cost())
            # print("betas = {}".format(self.plan.rots_coeffs))

        if self.plan.save_iterates:
            self.vol_iterates.append(self.plan.vol)
            self.rots_coeffs_iterates.append(self.plan.rots_coeffs)
            self.sigmas_iterates.append(self.plan.sigmas)
            self.tau_iterates.append(self.plan.tau)

    def finalize_solver(self):
        print("Solver has finished")

    def rots_density_step(self):
        n = self.plan.n
        eta = self.plan.eta
        lambd = self.plan.lambd
        dtype = self.plan.dtype

        self.plan.rots_coeffs = self.projection_simplex(
            - (n ** eta) / lambd * self.plan.data_discrepancy / (2 * self.plan.sigmas[None, :]), axis=0).astype(dtype)

    def sigma_step(self):
        self.plan.sigmas = np.sum(self.plan.data_discrepancy * self.plan.rots_coeffs, axis=0)

    def tau_step(self):
        self.plan.tau = np.sum(self.plan.vol.asnumpy() ** 2)

    def volume_step(self):  # TODO try to do same as rots projection, i.e., only use projections with non-zero coefficient
        L = self.plan.L
        n = self.plan.n
        dtype = self.plan.dtype

        rots_weights = (self.plan.tau / self.plan.sigmas[None, :]) * self.plan.rots_coeffs

        # compute adjoint forward map of images
        logger.info("Compute adjoint forward mapping on the images")
        src = self.plan.adjoint_forward(self.plan.images, rots_weights)
        # print("src = {}".format(src))

        # compute kernel in fourier domain
        _2L = 2 * L
        kernel = np.zeros((_2L, _2L, _2L), dtype=dtype)
        sq_filters_f = self.plan.eval_filter_grid(power=2)
        sq_filters_f *= self.plan.amplitude ** 2

        summed_rots_weights = np.sum(rots_weights, axis=1)

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
            # pts_rot = np.moveaxis(pts_rot, 1, 2)  # was in Aspire. Might be needed for non radial kernels
            pts_rot = m_reshape(pts_rot, (3, -1))

            kernel += (
                    1
                    / (L ** 6)
                    # / (L ** 4)
                    * anufft(weights, pts_rot, (_2L, _2L, _2L), real=True)
            )

        # print("kernel = {}".format(kernel))

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
                      / (L**3)  # Compensation for the lack of scaling in the forward operator
                      ).astype(dtype)

        self.plan.vol = Volume(vol)

        logger.info("Update data_discrepancies")
        self.plan.data_discrepancy_update()

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
