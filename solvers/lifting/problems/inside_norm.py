import logging

import numpy as np

from pymanopt.manifolds import Rotations

from aspire.image import Image
from aspire.volume import Volume

from solvers.lifting.integration import Integrator
from solvers.lifting.problems import LiftingProblem

logger = logging.getLogger(__name__)


class InsideNormLiftingProblem(LiftingProblem):
    def __init__(
            self,
            imgs=None,
            vol=None,
            filter=None,
            # offsets=None,
            amplitude=None,
            dtype=np.float32,
            integrator=None,
            rots_prior=None,
            seed=0,
            # memory=None, TODO Look into this once we get to larger data sets - use .npy files to save and load
    ):
        """
        A Cryo-EM lifting problem
        Other than the base class attributes, it has:
        :param imgs: A n-by-L-by-L array of noisy projection images (should not be confused with the .image objects)
        :param rots: A n-by-3-by-3 array of rotation matrices
        """

        super().__init__(imgs=imgs,
                         vol=vol,
                         filter=filter,
                         amplitude=amplitude,
                         dtype=dtype,
                         integrator=integrator,
                         seed=seed)

        # self.dtype = dtype
        #
        # if imgs is None:
        #     raise RuntimeError("No data provided")
        # else:
        #     assert isinstance(imgs, Image)
        #     self.imgs = imgs
        #
        # if vol is None:
        #     raise RuntimeError("No volume provided")
        # else:
        #     assert isinstance(vol, Volume)
        #     self.vol = vol
        #
        # if self.vol.dtype != self.dtype:
        #     logger.warning(
        #         f"{self.__class__.__name__}"
        #         f" vol.dtype {self.vol.dtype} != self.dtype {self.dtype}."
        #         " In the future this will raise an error."
        #     )
        #
        # self.L = imgs.shape[1]
        # self.N = imgs.shape[0]
        # self.dtype = np.dtype(dtype)
        #
        # self.seed = seed
        #
        # self.filter = filter
        #
        # # TODO we want to do this ourselves in the end
        # # if offsets is None:
        # #     offsets = np.zeros((2, self.n)).T
        # #
        # # self.offsets = offsets
        #
        # if amplitude is None:
        #     amplitude = 1.
        #
        # self.amplitude = amplitude
        #
        # if not isinstance(integrator, Integrator):
        #     raise RuntimeError("integrator is not an Integrator object")
        # self.integrator = integrator
        #
        # self.n = self.integrator.n
        # self.ell = self.integrator.ell
        #
        # if rots_prior is not None:
        #     logger.info("Computing rotation prior integrands")
        #     self.rots_prior_integrands = self.integrands_rots_prior(rots_prior)  # TODO check whether we can do this faster
        # else:
        #     self.rots_prior_integrands = None
        #
        # rot_dcoef = np.eye(1, self.integrator.ell)[0]
        # self.rots_dcoef = np.repeat(rot_dcoef[np.newaxis, :], self.N, axis=0)

    def integrands_forward(self):
        im = self.vol.project(0, self.integrator.rots)
        im = self.eval_filter(im)  # Here we only use 1 filter
        # im = im.shift(self.offsets[all_idx, :])  # TODO use this later on
        im *= self.amplitude  # [im.n, np.newaxis, np.newaxis]  # Here we only use 1 amplitude

        return im

    def forward(self):
        """
        Apply forward image model to volume
        :param vol: A volume instance.
        :param start: Start index of image to consider
        :param num: Number of images to consider
        :return: The images obtained from volume by projecting, applying CTFs, translating, and multiplying by the
            amplitude.
        """

        weights = self.integrator.coeffs2weights(self.rots_dcoef)
        integrands = self.integrands_forward().asnumpy()

        im = np.einsum("ij,jkl->ikl", weights, integrands).astype(self.dtype)

        return Image(im)

    def adjoint_forward(self, im):
        """
        Apply adjoint mapping to set of images
        :param im: An Image instance to which we wish to apply the adjoint of the forward model.
        :param start: Start index of image to consider
        :return: An L-by-L-by-L volume containing the sum of the adjoint mappings applied to the start+num-1 images.
        """

        # res = np.zeros((self.L, self.L, self.L), dtype=self.dtype)

        weights = self.integrator.coeffs2weights(self.rots_dcoef)

        integrands = Image(np.einsum("ij,ikl->jkl", weights, im.asnumpy()))
        integrands *= self.amplitude
        # im = im.shift(-self.offsets[all_idx, :])
        integrands = self.eval_filter(integrands)

        res = integrands.backproject(self.integrator.rots)[0]


        # weights = self.integrator.coeffs2weights(self.rots_dcoef).T
        #
        # # TODO can we speed this up somehow?
        # for g in range(self.n):
        #     weight_g = weights[g]
        #     # print("rot {} has total weight = {} and weight_g[g] = {}".format(g, np.sum(weight_g),weight_g[g]))
        #     im_g = im * weight_g[:, np.newaxis, np.newaxis]  # TODO hier kan het wel eens fout gaan
        #     im_g *= self.amplitude
        #     # im = im.shift(-self.offsets[all_idx, :])
        #     im_g = self.eval_filter(im_g)
        #
        #     rot = self.integrator.rots[g]
        #     rots = np.repeat(rot[np.newaxis, :, :], self.N, axis=0)
        #     integrand = im_g.backproject(rots)[0]
        #
        #     res += integrand.astype(self.dtype)

        logger.info(f"Determined adjoint mappings. Shape = {res.shape}")
        return res

    def eval_filter(self, im_orig):
        im = im_orig.copy()
        im = Image(im.asnumpy()).filter(self.filter)

        return im

    # def dens_forward(self, coeff):
    #     weights = self.integrator.coeffs2weights(coeff)
    #     integrands = self.integrands_vol_forward(self.vol).asnumpy()
    #
    #     im = np.einsum("ij,jkl->ikl", weights, integrands).astype(self.dtype)
    #
    #     return Image(im)
    #
    # def dens_adjoint_forward(self, im):
    #     integrands = self.integrands_vol_forward(self.vol).asnumpy()
    #
    #     weights = np.einsum("ijk,ljk->il", im.asnumpy(), integrands)  # TODO scale down with L**2?
    #     coeffs = self.integrator.weights2coeffs(weights)
    #
    #     return coeffs

    def integrands_rots_prior(self, rots, power=2):
        # TODO test
        manifold = Rotations(3)

        costs = np.zeros((self.N, self.n))

        total = self.N * self.n

        for i in range(self.N):
            rot_i = rots[i]
            for g in range(self.n):
                # rots_i = np.repeat(rot_i[np.newaxis, :, :], self.n, axis=0)  # TODO do we need this?
                cost = 1 / power * manifold.dist(rot_i, self.integrator.rots[g]) ** power
                costs[i, g] = cost

            logger.info("Progress: {} %".format(self.n*(i+1)/total * 100))

        return costs
