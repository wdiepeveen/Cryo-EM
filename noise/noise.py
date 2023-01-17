import numpy as np

from aspire.image import Image
from aspire.image.xform import Xform
from aspire.operators import PowerFilter, ZeroFilter, ScalarFilter
from aspire.utils.random import randn


class SnrNoiseAdder(Xform):
    """
    A Xform that adds white noise with constant SNR across all images,
    optionally passed through a Filter object, to all incoming images.
    """

    def __init__(self, seed=0, noise_filter=ScalarFilter(value=1.), snr=None):
        """
        Initialize the random state of this SnrNoiseAdder using specified values.
        :param seed: The random seed used to generate white noise
        :param noise_filter: An optional aspire.operators.Filter object to use to filter the generated white noise.
            By default, a ZeroFilter is used, generating no noise.
        """
        super().__init__()
        self.seed = seed
        noise_filter = noise_filter or ZeroFilter()
        self.noise_filter = noise_filter
        # self.noise_filter = PowerFilter(noise_filter, power=0.5)
        self.snr = snr

    def _forward(self, im, indices):
        im = im.copy()

        for i, idx in enumerate(indices):
            # Compute layer-wise SNR of im
            im_var = np.var(im.asnumpy()[i])

            random_seed = self.seed + 191 * (idx + 1)
            noisy_im = randn(im.res, im.res, seed=random_seed)
            noise_var = np.var(noisy_im)
            noisy_im = np.sqrt(im_var / (noise_var * self.snr)) * noisy_im

            noisy_im = Image(noisy_im).filter(self.noise_filter)[0]
            im[i] += noisy_im

            # im_s = randn(2 * im.res, 2 * im.res, seed=random_seed)
            #
            # im_s = Image(im_s).filter(self.noise_filter)[0]
            #
            # noisy_im = im_s[: im.res, : im.res]
            # noise_var = np.var(noisy_im)
            # noisy_im = np.sqrt(im_var / (noise_var * self.snr)) * noisy_im
            #
            # # im[i] += im_s[: im.res, : im.res]
            # im[i] += noisy_im

        return im
