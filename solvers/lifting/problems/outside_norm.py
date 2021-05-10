import numpy as np
import logging

from aspire.image import Image
from aspire.volume import Volume

from solvers.lifting.problems import LiftingProblem

logger = logging.getLogger(__name__)


class OutsideNormLiftingProblem(LiftingProblem):
    def __init__(self,
                 imgs=None,
                 vol=None,
                 filter=None,
                 # offsets=None,
                 amplitude=None,
                 dtype=np.float32,
                 integrator=None,
                 seed=0,
                 ):
        super().__init__(imgs=imgs,
                         vol=vol,
                         filter=filter,
                         amplitude=amplitude,
                         dtype=dtype,
                         integrator=integrator,
                         seed=seed)

    def forward(self):
        """
        Apply forward image model to volume
        :param vol: A volume instance.
        :param start: Start index of image to consider
        :param num: Number of images to consider
        :return: The images obtained from volume by projecting, applying CTFs, translating, and multiplying by the
            amplitude.
        """
        # print(type(self.vol))
        im = self.vol.project(0, self.integrator.rots)
        im = self.eval_filter(im)  # Here we only use 1 filter, but might as well do one for every entry
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
        weights = self.integrator.coeffs2weights(self.rots_dcoef)

        integrands = Image(np.einsum("ig,ikl->gkl", weights, im.asnumpy()))
        integrands *= self.amplitude
        # im = im.shift(-self.offsets[all_idx, :])
        integrands = self.eval_filter(integrands)

        res = integrands.backproject(self.integrator.rots)[0]

        logger.info(f"Determined adjoint mappings. Shape = {res.shape}")
        return res
