import logging

import numpy as np

from aspire.image import Image
from aspire.source import ImageSource
from aspire.utils.random import rand, randi
from aspire.volume import Volume

from solvers.lifting.integration import Integrator

logger = logging.getLogger(__name__)


class LiftingProblem:
    def __init__(
            self,
            imgs=None,
            vol=None,
            unique_filters=None,
            filter_indices=None,
            offsets=None,
            amplitudes=None,
            dtype=np.float32,
            integrator=None,
            lmax=5,
            rots_prior=None,
            seed=0,
            # memory=None, TODO Look into this once we get to larger data sets - use .npy files to save and load
    ):
        """
        A Cryo-EM primal-dual problem
        Other than the base class attributes, it has:
        :param imgs: A n-by-L-by-L array of noisy projection images (should not be confused with the .image objects)
        :param rots: A n-by-3-by-3 array of rotation matrices
        """

        if imgs is None:
            raise RuntimeError("No data provided")
        else:
            assert isinstance(imgs, Image)
            self.imgs = imgs

        if vol is None:
            raise RuntimeError("No volume provided")
        else:
            assert isinstance(vol, Volume)
            self.vol = vol

        if self.vol.dtype != self.dtype:
            logger.warning(
                f"{self.__class__.__name__}"
                f" vol.dtype {self.vol.dtype} != self.dtype {self.dtype}."
                " In the future this will raise an error."
            )

        self.L = imgs.shape[1]
        self.n = imgs.shape[0]
        self.dtype = np.dtype(dtype)

        self.seed = seed

        self.unique_filters = unique_filters

        # Create filter indices and fill the metadata based on unique filters
        if unique_filters:
            if filter_indices is None:
                filter_indices = np.zeros(self.n)
            self.filter_indices = filter_indices

        # TODO we want to do this ourselves in the end
        if offsets is None:
            offsets = np.zeros((2, self.n)).T

        self.offsets = offsets

        if amplitudes is None:
            amplitudes = np.ones(self.n)

        self.amplitudes = amplitudes

        if not isinstance(integrator, Integrator):
            raise RuntimeError("integrator is not an Integrator object")
        self.integrator = integrator

        self.rots_dcoef = None  # TODO make function that provides initial guess

        if rots_prior is not None:
            self.rots_prior = rots_prior

    def vol_forward(self, vol, rots, start=0, num=np.inf):
        """
        Apply forward image model to volume
        :param vol: A volume instance.
        :param start: Start index of image to consider
        :param num: Number of images to consider
        :return: The images obtained from volume by projecting, applying CTFs, translating, and multiplying by the
            amplitude.
        """
        all_idx = np.arange(start, min(start + num, self.n))
        assert vol.n_vols == 1, "vol_forward expects a single volume, not a stack"

        if vol.dtype != self.dtype:
            logger.warning(f"Volume.dtype {vol.dtype} inconsistent with {self.dtype}")

        im = vol.project(0, rots[all_idx, :, :])
        im = self.eval_filters(im, start, num)
        im = im.shift(self.offsets[all_idx, :])
        im *= self.amplitudes[all_idx, np.newaxis, np.newaxis]
        return im

    def im_adjoint_forward(self, im):
        """
        Apply adjoint mapping to source

        :return: The adjoint mapping applied to the images, averaged over the whole dataset and expressed
                    as coefficients of `basis`.
        """
        res = np.zeros((self.L, self.L, self.L), dtype=self.dtype)

        batch_mean_b = self.im_backward(im, 0) / self.n
        res += batch_mean_b.astype(self.dtype)

        logger.info(f"Determined adjoint mappings. Shape = {res.shape}")
        return res










