import numpy as np
import logging

from scipy.fftpack import fft2

from aspire.nufft import anufft
from aspire.reconstruction.kernel import FourierKernel
from aspire.utils.fft import mdim_ifftshift
from aspire.utils.matlab_compat import m_flatten, m_reshape
from aspire.volume import rotated_grids, Volume

from joblib import Parallel, delayed
import multiprocessing
from tqdm import tqdm

from projects.lifting_v2.src.plans.refinement_plan1 import Refinement_Plan1
from projects.lifting_v2.src.solvers import Joint_Volume_Rots_Solver

logger = logging.getLogger(__name__)


class Refinement_Solver1(Joint_Volume_Rots_Solver):
    def __init__(self,
                 vol=None,
                 rots=None,
                 squared_noise_level=None,
                 stop=None,
                 stop_rots_gd=None,
                 gd_step_size=10**-2,
                 gd_eta=0.25,
                 images=None,
                 filter=None,
                 amplitude=None,
                 kernel=None,
                 integrator=None,
                 volume_reg_param=None,
                 dtype=np.float32,
                 seed=0,
                 ):
        plan = Refinement_Plan1(vol=vol,
                                rots=rots,
                                stop=stop,  # TODO here a default stopping criterion
                                stop_rots_gd=stop_rots_gd,  # TODO here a default stopping criterion
                                gd_step_size=gd_step_size,
                                gd_eta=gd_eta,
                                squared_noise_level=squared_noise_level,
                                images=images,
                                filter=filter,
                                amplitude=amplitude,
                                kernel=kernel,
                                integrator=integrator,
                                volume_reg_param=volume_reg_param,
                                rots_batch_size=8192,
                                dtype=dtype,
                                seed=seed,
                                )

        super().__init__(plan=plan)

    def stop_solver(self):
        # TODO this one should probably go better elsewhere since it is quite default
        return self.iter == self.plan.o.stop  # TODO this now only works since we assume that this is an integer

    def step_solver(self):
        self.rots_step()
        # self.cost.append(self.plan.get_cost())

        self.volume_step()
        # self.cost.append(self.plan.get_cost())

    def finalize_solver(self):
        print("Solver has finished")

    def rots_step(self):

        def stop_gradient_descent(iter):  # TODO make a proper option our of this which we can feed into the class
            return iter == self.plan.o.stop_rots_gd

        print("Start Riemannian Gradient Descent Solver")
        Costs = []
        Relerrors = []
        quaternions = self.plan.quaternions

        def do_gradient_descent(iter):
            k = 0
            quat = quaternions[iter, :]
            cost = self.plan.get_cost(index=iter)
            cost0 = cost
            # costs = [cost]
            gradient = self.compute_Riemannian_gradient(index=iter)
            logger.debug("gradient = {}".format(gradient))
            normgradient0 = np.linalg.norm(gradient)
            relerror = 1.
            # relerrors = [relerror]
            logger.info("==================== Image {} ====================".format(iter + 1))
            # print("==================== Image {} ====================".format(iter + 1))
            logger.info("{} | k = {} | relative cost = {} | relerror = {} | ||gradient|| = {}".format(iter + 1, k, 1., relerror, normgradient0))
            while not stop_gradient_descent(k) and relerror > 1e-3:
                # Use resulting quat and gradient computed from previous iteration
                logger.debug("quat = {}".format(quat))
                logger.debug("gradient = {}".format(gradient))
                step_size = self.plan.o.gd_step_size
                new_quat = self.plan.p.manifold.exp(quat, - step_size * gradient)
                new_cost = self.plan.get_cost(index=iter, quaternion=new_quat)
                j = 0
                break_out_flag = False
                while new_cost > cost:
                    if j >= 10:
                        break_out_flag = True
                        break
                    step_size *= self.plan.o.gd_eta
                    new_quat = self.plan.p.manifold.exp(quat, - step_size * gradient)
                    new_cost = self.plan.get_cost(index=iter, quaternion=new_quat)
                    j += 1

                if break_out_flag:
                    logger.info(
                        "{} | k = {} | broken out".format(iter + 1, k + 1))
                    break

                cost = new_cost
                # costs.append(cost)

                quat = new_quat
                gradient = self.compute_Riemannian_gradient(index=iter, quaternion=new_quat)

                normgradient = np.linalg.norm(gradient)
                relerror = normgradient / normgradient0
                # relerrors.append(relerror)
                k += 1
                if k%5 != 0:
                    logger.debug(
                        "{} | k = {} | relative cost = {} | relative error = {} | ||gradient|| = {}".format(iter + 1, k,
                                                                                                            cost / cost0,
                                                                                                            relerror,
                                                                                                            normgradient))
                else:
                    logger.info(
                        "{} | k = {} | relative cost = {} | relative error = {} | ||gradient|| = {}".format(iter + 1, k, cost / cost0, relerror, normgradient))

            return quat

        if not stop_gradient_descent(0):
            num_cores = multiprocessing.cpu_count()
            logger.info("num_cores = {}".format(num_cores))
            results = Parallel(n_jobs=num_cores)(delayed(do_gradient_descent)(i) for i in tqdm(range(self.plan.p.N)))  # TODO 4 Nov 2021
            quaternions = np.vstack(results)

            logger.info("result.shape = {}".format(quaternions.shape))
            logger.info("result = {}".format(quaternions))
            # for i in range(self.plan.p.N):  # TODO parallellize
                # # TODO test for loop then write function out of this and parallellize
                # k = 0
                # cost = self.plan.get_cost(index=i)
                # cost0 = cost
                # costs = [cost]
                # gradient = self.compute_Riemannian_gradient(index=i)
                # logger.debug("gradient = {}".format(gradient))
                # normgradient0 = np.linalg.norm(gradient)
                # relerror = 1.
                # relerrors = [relerror]
                # logger.info("==================== Image {} ====================".format(i+1))
                # logger.info("{} | k = {} | relative cost = {} | relerror = {}".format(i+1, k, 1., relerror))
                # while not stop_gradient_descent(k) and relerror > 1e-3:
                #     quat = quaternions[i, :]
                #     logger.debug("quat = {}".format(quat))
                #
                #     # Use gradient computed in previous iteration
                #     logger.debug("gradient = {}".format(gradient))
                #     step_size = self.plan.o.gd_step_size
                #     new_quat = self.plan.p.manifold.exp(quat, - step_size * gradient)
                #     new_cost = self.plan.get_cost(index=i, quaternion=new_quat)
                #     j = 0
                #     break_out_flag = False
                #     while new_cost > cost:
                #         if j>=10:
                #             break_out_flag = True
                #             break
                #         step_size *= self.plan.o.gd_eta
                #         new_quat = self.plan.p.manifold.exp(quat, - step_size * gradient)
                #         new_cost = self.plan.get_cost(index=i, quaternion=new_quat)
                #         j+=1
                #
                #     if break_out_flag:
                #         logger.info(
                #             "{} | k = {} | broken out".format(i + 1, k + 1))
                #         break
                #
                #     cost = new_cost
                #     costs.append(cost)
                #
                #     quaternions[i, :] = new_quat
                #     gradient = self.compute_Riemannian_gradient(index=i, quaternion=new_quat)
                #
                #     relerror = np.linalg.norm(gradient)/normgradient0
                #     relerrors.append(relerror)
                #     k += 1
                #     logger.info(
                #         "{} | k = {} | relative cost = {} | relative error = {}".format(i + 1, k, cost / cost0, relerror))
                #
                #
                # Costs.append(costs)
                # Relerrors.append(relerrors)

        self.plan.quaternions = quaternions

    def compute_Riemannian_gradient(self, index=None, quaternion=None, rescale=True):
        assert index is not None

        if quaternion is None:
            quaternion = self.plan.quaternions[index, :]

        # Construct sampling sets for all images
        integrator = self.plan.p.integrator.update(quaternion=quaternion)

        # Compute data fidelity terms \|Ag.u - fi\|^2:
        # logger.info("Computing qs")
        im = self.plan.p.images.asnumpy()[index, :, :]

        rots_sampling_projections = self.plan.forward(self.plan.vol,
                                                      integrator.rots).asnumpy()  # TODO check whether we get correct rot input here

        q1 = np.sum(rots_sampling_projections ** 2, axis=(1, 2))
        q2 = - 2 * np.einsum("jk,gjk->g", im, rots_sampling_projections)
        q3 = np.sum(im ** 2)

        data_fidelity = ((q1 + q2 + q3) / (2 * self.plan.o.squared_noise_level * self.plan.p.L ** 2)).astype(
            self.plan.p.dtype)
        # print(data_fidelity)

        # Compute gradients (n, 4)
        gradients = self.plan.p.kernel.gradient(free_quaternion=quaternion[None, None, :],
                                                fixed_quaternion=integrator.quaternions[None, :, :])  # .swapaxes(0, 1))
        # print(gradients)

        # Reduce gradients (4)
        gradient = np.einsum("g,gj->j", data_fidelity, gradients) / integrator.n  # TODO use integrate function here?
        if rescale:
            gradient /= self.plan.p.kernel.norm**2

        # print(gradient)
        return gradient

    def volume_step(self):
        L = self.plan.p.L
        N = self.plan.p.N
        dtype = self.plan.p.dtype

        # compute adjoint forward map of images
        logger.info("Compute adjoint forward mapping on the images")
        src = self.plan.adjoint_forward(self.plan.p.images)

        # compute kernel in fourier domain
        _2L = 2 * L
        kernel = np.zeros((_2L, _2L, _2L), dtype=dtype)
        sq_filters_f = self.plan.eval_filter_grid(power=2)
        sq_filters_f *= self.plan.p.amplitude ** 2

        for start in range(0, N, self.plan.o.rots_batch_size):
            logger.info(
                "Running through projections {}/{} = {}%".format(start, N, np.round(start / N * 100, 2)))
            all_idx = np.arange(start, min(start + self.plan.o.rots_batch_size, N))
            num_idx = len(all_idx)

            weights = np.repeat(sq_filters_f[:, :, None], num_idx, axis=2)

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

        self.plan.vol = Volume(vol)

    # def compute_Riemannian_gradient(self):
    #     L = self.plan.p.L
    #     N = self.plan.p.N
    #     n = self.plan.p.n
    #     dtype = self.plan.p.dtype
    #
    #     # Construct sampling sets for all images
    #     # multi_integrator = Local_Regular(quaternions=self.plan.o.quaternions, dtype=dtype)  # TODO l and sep_dist
    #     multi_integrator = self.plan.p.integrator.update(quaternions=self.plan.o.quaternions)
    #     rots = multi_integrator.rots  # (n, N, 3, 3)
    #
    #     # Compute data fidelity terms \|Ag.u - fi\|^2:
    #     logger.info("Computing qs")
    #     im = self.plan.p.images.asnumpy()
    #     qs = np.zeros((n, N), dtype=self.plan.p.dtype)
    #     logger.info("Construct qs with batch size {}".format(self.plan.o.rots_batch_size))
    #     q3 = np.sum(im ** 2, axis=(1, 2))
    #     img_batch_size = int(self.plan.o.rots_batch_size / n)
    #     assert img_batch_size > 0
    #
    #     for start in range(0, N, img_batch_size):
    #         logger.info("Running through images {}/{} = {}%".format(start, N, np.round(start / N * 100, 2)))
    #         all_idx = np.arange(start, min(start + img_batch_size, N))
    #         num_idx = len(all_idx)
    #         selected_rots = rots[:, all_idx, :, :].reshape((-1, 3, 3))
    #         rots_sampling_projections = self.plan.forward(self.plan.o.vol, selected_rots).asnumpy().reshape(n, num_idx,
    #                                                                                                         L, L)
    #
    #         q1 = np.sum(rots_sampling_projections ** 2, axis=(2, 3))
    #         q2 = - 2 * np.einsum("ijk,gijk->gi", im[all_idx, :, :], rots_sampling_projections)
    #
    #         qs[:, all_idx] = (q1 + q2 + q3[:, all_idx]) / (2 * self.plan.o.squared_noise_level * L ** 2)
    #
    #     # Compute gradients (n, N, 4)
    #     gradients = self.plan.p.kernel.gradient(free_quaternion=self.plan.o.quaternions[:, None, :],
    #                                             fixed_quaternion=multi_integrator.quaternions.swapaxes(0, 1)).swapaxes(
    #         0, 1)
    #
    #     # Reduce gradients (N, 4)
    #     gradient = np.einsum("gi,gij->ij", qs, gradients) / n
    #     return gradient
