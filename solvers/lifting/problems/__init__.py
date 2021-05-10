import logging

import numpy as np

from pymanopt.manifolds import Rotations

from aspire.image import Image
from aspire.volume import Volume

from solvers.lifting.integration import Integrator

logger = logging.getLogger(__name__)


class LiftingProblem:
    def __init__(
            self,
            imgs=None,
            vol=None,
            filter=None,
            # offsets=None,
            amplitude=None,
            dtype=np.float32,
            integrator=None,
            seed=0,
            # memory=None, TODO Look into this once we get to larger data sets - use .npy files to save and load
    ):
        """
        A Cryo-EM lifting problem
        Other than the base class attributes, it has:
        :param imgs: A n-by-L-by-L array of noisy projection images (should not be confused with the .image objects)
        """

        self.dtype = dtype

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
        self.N = imgs.shape[0]
        self.dtype = np.dtype(dtype)

        self.seed = seed

        self.filter = filter

        # TODO we want to do this ourselves in the end
        # if offsets is None:
        #     offsets = np.zeros((2, self.n)).T
        #
        # self.offsets = offsets

        if amplitude is None:
            amplitude = 1.

        self.amplitude = amplitude

        if not isinstance(integrator, Integrator):
            raise RuntimeError("integrator is not an Integrator object")
        self.integrator = integrator

        self.n = self.integrator.n
        self.ell = self.integrator.ell

        rot_dcoef = np.eye(1, self.ell)[0]
        self.rots_dcoef = np.repeat(rot_dcoef[np.newaxis, :], self.N, axis=0)

    def forward(self):
        """
        Apply forward image model to volume
        :param vol: A volume instance.
        :param start: Start index of image to consider
        :param num: Number of images to consider
        :return: The images obtained from volume by projecting, applying CTFs, translating, and multiplying by the
            amplitude.
        """
        raise NotImplementedError(
            "Subclasses should implement this and return an Image object"
        )

    def adjoint_forward(self, im):
        """
        Apply adjoint mapping to set of images
        :param im: An Image instance to which we wish to apply the adjoint of the forward model.
        :param start: Start index of image to consider
        :return: An L-by-L-by-L volume containing the sum of the adjoint mappings applied to the start+num-1 images.
        """
        raise NotImplementedError(
            "Subclasses should implement this and return an Volume object"
        )

    def eval_filter(self, im_orig):
        im = im_orig.copy()
        im = Image(im.asnumpy()).filter(self.filter)

        return im
