import numpy as np
import logging

from aspire.image import Image
from aspire.utils.coor_trans import grid_2d
from aspire.volume import Volume

from projects.rkhs_lifting.src.plans import Plan

logger = logging.getLogger(__name__)


class Lifting_Plan1(Plan):
    """Class for preprocessing inputs and defining several functions such as cost and forward operator"""

    def __init__(self,
                 vol=None,
                 rots_coeffs=None,
                 max_iter=None,
                 squared_noise_level=None,
                 images=None,
                 filter=None,
                 amplitude=None,
                 integrator=None,
                 volume_reg_param=None,
                 rots_reg_param=None,
                 rots_coeffs_reg_param_rate=1,
                 rots_coeffs_reg_scaling_param=None,
                 rots_batch_size=8192,
                 save_iterates=False,
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
        self.squared_noise_level = squared_noise_level

        self.L = images.shape[1]
        self.N = images.shape[0]

        self.seed = seed

        self.filter = filter

        if amplitude is None:
            amplitude = 1.

        self.amplitude = amplitude

        self.integrator = integrator
        self.n = self.integrator.n
        self.nn = self.integrator.nn  # TODO add this function in kernel integrator

        self.lam1 = volume_reg_param
        if type(rots_reg_param) is tuple:
            assert len(rots_reg_param)==2
            self.lam2_init =rots_reg_param[0]
            self.lam2_inf = rots_reg_param[1]
            self.lam2 = self.lam2_init
            self.lam2_rate = rots_coeffs_reg_param_rate
        else:
            self.lam2_init = rots_reg_param
            self.lam2_inf = rots_reg_param
            self.lam2 = rots_reg_param
            self.lam2_rate = rots_coeffs_reg_param_rate
        self.p = rots_coeffs_reg_scaling_param

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
        self.squared_noise_level = squared_noise_level

        self.rots_coeffs = rots_coeffs

        self.rots_batch_size = rots_batch_size

        self.max_iter = max_iter
        self.save_iterates = save_iterates

    def get_cost(self):
        # Compute q's
        im = self.images.asnumpy()
        qs = np.zeros((self.n, self.N), dtype=self.dtype)
        logger.info("Construct qs with batch size {}".format(self.rots_batch_size))
        q3 = np.sum(im ** 2, axis=(1, 2))[None, :]
        for start in range(0, self.n, self.rots_batch_size):
            logger.info("Running through projections {}/{} = {}%".format(start, self.n, np.round(start/self.n*100,2)))
            rots_sampling_projections = self.forward(self.vol, start, self.rots_batch_size).asnumpy()

            q1 = np.sum(rots_sampling_projections ** 2, axis=(1, 2))[:, None]
            q2 = - 2 * np.einsum("ijk,gjk->gi", im, rots_sampling_projections)

            all_idx = np.arange(start, min(start + self.rots_batch_size, self.n))
            qs[all_idx, :] = (q1 + q2 + q3) / (2 * self.squared_noise_level * self.L ** 2)


        rhos = self.integrator.coeffs_to_weights(self.rots_coeffs)
        data_fidelity_penalty = np.sum(qs * rhos)

        vol_l2_penalty = self.lam1 / (2 * self.L ** 3) * np.sum(self.vol.asnumpy() ** 2)  # TODO factor 2L instead of just L?

        dens_l2_penalty = self.lam2 * self.n / 2 * np.sum(self.rots_coeffs ** 2)

        cost = data_fidelity_penalty + vol_l2_penalty + dens_l2_penalty

        logger.info(
            "data penalty = {} | vol_reg penalty = {} | dens_reg1 penalty = {}".format(
                data_fidelity_penalty,
                vol_l2_penalty,
                dens_l2_penalty)
        )
        return cost

    def get_solver_result(self):
        return self.vol

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
        im = self.eval_filter(im)  # Here we only use 1 filter, but might as well do one for every entry
        # im = im.shift(self.offsets[all_idx, :])  # TODO use this later on
        im *= self.amplitude  # [im.n, np.newaxis, np.newaxis]  # Here we only use 1 amplitude,
        # but might as well do one for every entry

        return im

    def adjoint_forward(self, im, weights):
        """
        Apply adjoint mapping to set of images
        :param im: An Image instance to which we wish to apply the adjoint of the forward model.
        :param start: Start index of image to consider
        :return: An L-by-L-by-L volume containing the sum of the adjoint mappings applied to the start+num-1 images.
        """
        res = np.zeros((self.L, self.L, self.L), dtype=self.dtype)
        for start in range(0, self.nn, self.rots_batch_size):
            logger.info(
                "Running through projections {}/{} = {}%".format(start, self.nn, np.round(start / self.nn * 100, 2)))
            all_idx = np.arange(start, min(start + self.rots_batch_size, self.nn))

            integrands = Image(np.einsum("gi,ikl->gkl", weights[all_idx, :], im.asnumpy()))
            integrands *= self.amplitude
            # im = im.shift(-self.offsets[all_idx, :])
            integrands = self.eval_filter(integrands)

            res += integrands.backproject(self.integrator.rotsrots[all_idx, :, :])[0]

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
