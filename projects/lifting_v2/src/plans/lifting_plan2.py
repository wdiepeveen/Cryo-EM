import numpy as np
import logging

from aspire.image import Image
from aspire.utils.coor_trans import grid_2d
from aspire.volume import Volume

from projects.lifting_v2.src.plans import Plan

logger = logging.getLogger(__name__)


class Lifting_Plan2(Plan):
    """Class for preprocessing inputs and defining several functions such as cost and forward operator"""

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
                 rots_batch_size=8192,
                 dtype=np.float32,
                 seed=0,
                 ):

        # Initialize Problem

        if images is None:
            raise RuntimeError("No data provided")
        else:
            assert isinstance(images, Image)

        self.dtype = dtype

        self.images = images

        self.L = images.shape[1]
        self.N = images.shape[0]

        self.seed = seed

        self.filter = filter

        if amplitude is None:
            amplitude = 1.

        self.amplitude = amplitude

        self.integrator = integrator
        self.n = self.integrator.n

        self.lambd = rots_reg_param
        self.eta = rots_reg_scaling_param

        # Initialize Options

        if vol is None:
            raise RuntimeError("No volume provided")
        else:
            assert isinstance(vol, Volume)

        if vol.dtype != self.dtype:
            logger.warning(
                f"{self.__class__.__name__}"
                f" vol.dtype {vol.dtype} != self.dtype {self.dtype}."
                " In the future this will raise an error."
            )

        # Initialize density coefficients
        if rots_coeffs is None:
            logger.info("Initializing density")
            rots_coeffs = 1 / self.n * np.ones((self.n, self.N), dtype=self.dtype)

        self.vol = vol
        self.rots_coeffs = rots_coeffs
        self.sigmas = squared_noise_level * np.ones((self.N,))
        self.tau = volume_reg_param

        self.data_discrepancy = np.zeros((self.n, self.N))  # (\|Ag.u - f_i\|^2)_g,i

        self.rots_batch_size = rots_batch_size

        self.max_iter = max_iter
        self.save_iterates = save_iterates

    def get_cost(self):
        data_term = np.sum(1 / (2 * self.sigmas[None, :]) * self.data_discrepancy * self.rots_coeffs)
        vol_reg_term = 1 / (2 * self.tau) * np.sum(self.vol.asnumpy() ** 2)   # / self.L ** 3
        rot_reg_term = self.lambd / (2 * self.n ** self.eta) * np.sum(self.rots_coeffs ** 2)
        sigmas_reg_term = 1/2 * np.sum(np.log(self.sigmas))
        tau_reg_term = 1/2 * np.log(self.tau)

        cost = data_term + vol_reg_term + rot_reg_term + sigmas_reg_term + tau_reg_term
        # TODO split costs
        return cost

    def get_solver_result(self):
        return self.vol

    def data_discrepancy_update(self):
        L = self.L
        N = self.N
        n = self.n
        dtype = self.dtype

        logger.info("Computing \|Ag.u - f_i\|^2")
        im = self.images.asnumpy()
        F = np.zeros((n, N), dtype=dtype)
        F3 = np.sum(im ** 2, axis=(1, 2))[None, :]
        # print("F3 = {}".format(F3[:, 0]))

        for start in range(0, n, self.rots_batch_size):
            all_idx = np.arange(start, min(start + self.rots_batch_size, n))
            logger.info(
                "Computing data fidelity at {}%".format(int((all_idx[-1] + 1) / n * 100)))

            rots_sampling_projections = self.forward(self.vol, start, self.rots_batch_size).asnumpy()

            F1 = np.sum(rots_sampling_projections ** 2, axis=(1, 2))[:, None]
            # print("F1 = {}".format(F1[:, 0]))
            F2 = - 2 * np.einsum("ijk,gjk->gi", im, rots_sampling_projections)
            # print("F2 = {}".format(F2[:, 0]))

            F[all_idx, :] = (F1 + F2 + F3)  # / (L ** 2)  # 2 * self.plan.squared_noise_level missing now

        self.data_discrepancy = F
        # print("F = {}".format(F[:, 0]))

    def forward(self, vol, start, num):
        """
        Apply forward image model to volume
        :param vol: A volume instance.
        :param start: Start index of image to consider
        :param num: Number of images to consider
        :return: The images obtained from volume by projecting, applying CTFs, translating, and multiplying by the
            amplitude.
        """
        all_idx = np.arange(start, min(start + num, self.n))
        im = vol.project(0, self.integrator.rots[all_idx, :, :])
        # im *= 1 / self.L  # Rescale for projection
        im = self.eval_filter(im)  # Here we only use 1 filter, but might as well do one for every entry
        # im *= 1 / (self.L ** 2)  # Rescale for FT
        # im = im.shift(self.offsets[all_idx, :])  # TODO use this later on
        im *= self.amplitude  # [im.n, np.newaxis, np.newaxis]  # Here we only use 1 amplitude,
        # but might as well do one for every entry

        return im

    def adjoint_forward(self, im):
        """
        Apply adjoint mapping to set of images
        :param im: An Image instance to which we wish to apply the adjoint of the forward model.
        :param start: Start index of image to consider
        :return: An L-by-L-by-L volume containing the sum of the adjoint mappings applied to the start+num-1 images.
        """
        quat_idx = np.arange(0, self.n)[np.sum(self.rots_coeffs, axis=1) > 0.]  # Is any of the rotations unused?
        weights = (self.tau / self.sigmas[None, :]) * self.rots_coeffs

        res = np.zeros((self.L, self.L, self.L), dtype=self.dtype)
        for start in range(0, len(quat_idx), self.rots_batch_size):
            all_idx = np.arange(start, min(start + self.rots_batch_size, len(quat_idx)))
            logger.info(
                "Computing adjoint forward mappings at {}%".format(int((all_idx[-1] + 1) / len(quat_idx) * 100)))

            idx = quat_idx[all_idx]
            integrands = Image(np.einsum("gi,ikl->gkl", weights[idx, :], im.asnumpy()))
            integrands *= self.amplitude
            # im = im.shift(-self.offsets[all_idx, :])
            integrands = self.eval_filter(integrands)

            res += integrands.backproject(self.integrator.rots[idx, :, :])[0]

        logger.info(f"Determined adjoint mappings. Shape = {res.shape}")
        return res

    def eval_filter(self, im_orig):
        im = im_orig.copy()
        im = Image(im.asnumpy()).filter(self.filter)

        return im

    def eval_filter_grid(self, power=1):
        dtype = self.dtype
        L = self.L

        grid2d = grid_2d(L, dtype=dtype)
        omega = np.pi * np.vstack((grid2d["x"].flatten(), grid2d["y"].flatten()))

        filter_values = self.filter.evaluate(omega)
        if power != 1:
            filter_values **= power

        h = np.reshape(filter_values, grid2d["x"].shape)

        return h
