import numpy as np
import logging

from aspire.image import Image
from aspire.utils.coor_trans import grid_2d
from aspire.volume import Volume

from projects.rkhs_lifting.src.plans import Plan
from projects.rkhs_lifting.src.plans.problems.lifting_problem1 import Lifting_Problem1
from projects.rkhs_lifting.src.plans.options.lifting_options1 import Lifting_Options1

logger = logging.getLogger(__name__)


class Lifting_Plan1(Plan):
    """Class for preprocessing inputs and defining several functions such as cost and forward operator"""

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
                 batch_size=8192,
                 dtype=np.float32,
                 seed=0,
                 ):

        if images is None:
            raise RuntimeError("No data provided")
        else:
            assert isinstance(images, Image)

        self.p = Lifting_Problem1(images=images,
                                  filter=filter,
                                  amplitude=amplitude,
                                  integrator=integrator,
                                  volume_reg_param=volume_reg_param,
                                  rots_density_reg_param=rots_density_reg_param,
                                  dtype=dtype,
                                  seed=seed)

        if vol is None:
            raise RuntimeError("No volume provided")
        else:
            assert isinstance(vol, Volume)

        if vol.dtype != self.p.dtype:
            logger.warning(
                f"{self.__class__.__name__}"
                f" vol.dtype {vol.dtype} != self.dtype {self.p.dtype}."
                " In the future this will raise an error."
            )

        # Initialize density coefficients
        if density_coeffs is None:
            logger.info("Initializing density")
            im = self.p.images.asnumpy()
            qs = np.zeros((self.p.n, self.p.N), dtype=self.p.dtype)
            logger.info("Construct qs with batch size {}".format(batch_size))
            q3 = np.sum(im ** 2, axis=(1, 2))[None, :]
            for start in range(0, self.p.n, batch_size):
                logger.info("Running through projections {}/{} = {}%".format(start, self.p.n,
                                                                             np.round(start / self.p.n * 100, 2)))
                rots_sampling_projections = self.forward(vol, start, batch_size).asnumpy()

                q1 = np.sum(rots_sampling_projections ** 2, axis=(1, 2))[:, None]
                q2 = - 2 * np.einsum("ijk,gjk->gi", im, rots_sampling_projections)

                all_idx = np.arange(start, min(start + batch_size, self.p.n))
                qs[all_idx, :] = (q1 + q2 + q3) / (2 * squared_noise_level * self.p.L ** 2)

            logger.info("Start computing Wqs")
            Wqs = self.p.integrator.coeffs_to_weights(qs)
            argmins = np.argmin(Wqs, axis=0)
            # print(argmaxes)
            density_coeffs = np.zeros((self.p.n, self.p.N), dtype=self.p.dtype)
            density_coeffs[argmins,np.arange(self.p.N)] = 1

            # density_coeffs = 1 / self.p.n * np.ones((self.p.n, self.p.N), dtype=self.p.dtype)

        if dual_coeffs is None:
            dual_coeffs = np.zeros((1, self.p.N), dtype=self.p.dtype)

        self.o = Lifting_Options1(vol=vol,
                                  squared_noise_level=squared_noise_level,
                                  density_coeffs=density_coeffs,
                                  dual_coeffs=dual_coeffs,
                                  stop=stop,
                                  stop_density_update=stop_density_update,
                                  batch_size=batch_size,
                                  )

    def get_cost(self):
        # Compute q's
        im = self.p.images.asnumpy()
        qs = np.zeros((self.p.n, self.p.N), dtype=self.p.dtype)
        logger.info("Construct qs with batch size {}".format(self.o.batch_size))
        q3 = np.sum(im ** 2, axis=(1, 2))[None, :]
        for start in range(0, self.p.n, self.o.batch_size):
            logger.info(
                "Running through projections {}/{} = {}%".format(start, self.p.n, np.round(start / self.p.n * 100, 2)))
            rots_sampling_projections = self.forward(self.o.vol, start, self.o.batch_size).asnumpy()

            q1 = np.sum(rots_sampling_projections ** 2, axis=(1, 2))[:, None]
            q2 = - 2 * np.einsum("ijk,gjk->gi", im, rots_sampling_projections)

            all_idx = np.arange(start, min(start + self.o.batch_size, self.p.n))
            qs[all_idx, :] = (q1 + q2 + q3) / (2 * self.o.squared_noise_level * self.p.L ** 2)

        rhos = self.p.integrator.coeffs_to_weights(self.o.density_coeffs)
        data_fidelity_penalty = np.sum(qs * rhos)

        vol_l2_penalty = self.p.volume_reg_param / (2 * self.p.L ** 3) * np.sum(
            self.o.vol.asnumpy() ** 2)  # TODO factor 2L instead of just L?

        dens_l2_penalty = self.p.rots_density_reg_param * self.p.n / 2 * np.sum(self.o.density_coeffs ** 2)

        cost = data_fidelity_penalty + vol_l2_penalty + dens_l2_penalty

        logger.info(
            "data penalty = {} | vol_reg penalty = {} | dens_reg1 penalty = {}".format(
                data_fidelity_penalty,
                vol_l2_penalty,
                dens_l2_penalty)
        )
        return cost

    def forward(self, vol, start, num):
        """
        Apply forward image model to volume
        :param vol: A volume instance.
        :param start: Start index of image to consider
        :param num: Number of images to consider
        :return: The images obtained from volume by projecting, applying CTFs, translating, and multiplying by the
            amplitude.
        """
        all_idx = np.arange(start, min(start + num, self.p.n))
        # print(type(self.vol))
        im = vol.project(0, self.p.integrator.rots[all_idx, :, :])
        im = self.eval_filter(im)  # Here we only use 1 filter, but might as well do one for every entry
        # im = im.shift(self.offsets[all_idx, :])  # TODO use this later on
        im *= self.p.amplitude  # [im.n, np.newaxis, np.newaxis]  # Here we only use 1 amplitude,
        # but might as well do one for every entry

        return im

    def adjoint_forward(self, im):
        """
        Apply adjoint mapping to set of images
        :param im: An Image instance to which we wish to apply the adjoint of the forward model.
        :param start: Start index of image to consider
        :return: An L-by-L-by-L volume containing the sum of the adjoint mappings applied to the start+num-1 images.
        """
        weights = self.p.integrator.coeffs_to_weights(self.o.density_coeffs)

        integrands = Image(np.einsum("gi,ikl->gkl", weights, im.asnumpy()))
        integrands *= self.p.amplitude
        # im = im.shift(-self.offsets[all_idx, :])
        integrands = self.eval_filter(integrands)
        # TODO here we need an iteration over all batches for the backproject part

        res = integrands.backproject(self.p.integrator.rots)[0]

        logger.info(f"Determined adjoint mappings. Shape = {res.shape}")
        return res

    def eval_filter(self, im_orig):
        im = im_orig.copy()
        im = Image(im.asnumpy()).filter(self.p.filter)

        return im

    def eval_filter_grid(self, power=1):
        dtype = self.p.dtype
        L = self.p.L

        grid2d = grid_2d(L, dtype=dtype)
        omega = np.pi * np.vstack((grid2d["x"].flatten(), grid2d["y"].flatten()))

        filter_values = self.p.filter.evaluate(omega)
        if power != 1:
            filter_values **= power

        h = np.reshape(filter_values, grid2d["x"].shape)

        return h
