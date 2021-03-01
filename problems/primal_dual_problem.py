import logging

import numpy as np

from aspire.image import Image
from aspire.image.xform import NoiseAdder
from aspire.operators import ZeroFilter
from aspire.source import ImageSource
from aspire.utils.random import Random, rand, randi, randn
from aspire.utils.coor_trans import uniform_random_angles
from aspire.volume import Volume

logger = logging.getLogger(__name__)


class PrimalDualProblem(ImageSource):
    def __init__(
            self,
            data=None,
            vols=None,
            states=None, #no right?
            unique_filters=None,
            filter_indices=None,
            offsets=None,
            amplitudes=None,
            dtype=np.float32,
            rots=None,
            rots_prior=None,
            seed=0,
            memory=None,
    ):
        """
        A Cryo-EM primal-dual problem
        Other than the base class attributes, it has:
        :param data: A n-by-L-by-L array of noisy projection images (should not be confused with the .image objects)
        :param rots: A n-by-3-by-3 array of rotation matrices
        """

        if data is None:
            raise RuntimeError("No data provided")
        else:
            assert isinstance(data, Image)
            self.data = data

        super().__init__(L=data.shape[1], n=data.shape[0], dtype=dtype, memory=memory)

        self.seed = seed

        # We need to keep track of the original resolution we were initialized with,
        # to be able to generate projections of volumes later, when we are asked to supply images.
        self._original_L = self.L

        # TODO undo this -> we want to do this ourselves in the end
        if offsets is None:
            offsets = np.zeros((2, self.n)).T
            # offsets = self.L / 16 * randn(2, self.n, seed=seed).astype(dtype).T
        # TODO check why we need this/where it is used
        if amplitudes is None:
            min_, max_ = 2.0 / 3, 3.0 / 2
            amplitudes = min_ + rand(self.n, seed=seed).astype(dtype) * (max_ - min_)

        if vols is None:
            raise RuntimeError("No volume provided")
        else:
            assert isinstance(vols, Volume)
            self.vols = vols
            self.init_vol = vols

        if self.vols.dtype != self.dtype:
            logger.warning(
                f"{self.__class__.__name__}"
                f" vols.dtype {self.vols.dtype} != self.dtype {self.dtype}."
                " In the future this will raise an error."
            )

        self.C = self.vols.n_vols

        # TODO remove states: In this solver these aren't used and must remain unknown
        self.states = states or randi(self.C, self.n, seed=seed)

        if rots is None:
            raise RuntimeError("No rotations provided")
        else:
            self.rots = rots

        # rotations we will need as a guess for the regularizer
        if rots_prior is None:
            self.init_rots = rots
        else:
            self.init_rots = rots_prior

        self.unique_filters = unique_filters

        # Create filter indices and fill the metadata based on unique filters
        if unique_filters:
            if filter_indices is None:
                filter_indices = randi(len(unique_filters), self.n, seed=seed) - 1
            self.filter_indices = filter_indices

        self.offsets = offsets
        self.amplitudes = amplitudes


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








