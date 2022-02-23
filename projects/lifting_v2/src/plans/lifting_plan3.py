import numpy as np
import logging

import quaternionic

from scipy.spatial.transform import Rotation as R

from aspire.image import Image
from aspire.utils.coor_trans import grid_2d
from aspire.volume import Volume

from projects.lifting_v2.src.plans import Plan

logger = logging.getLogger(__name__)


class Lifting_Plan3(Plan):
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
                 J0=None,
                 # solver options
                 max_iter=None,
                 save_iterates=False,
                 rots_batch_size=1024,
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
        self._points = None
        self.sigmas = squared_noise_level * np.ones((self.N,))
        self.tau = volume_reg_param

        if (rots_reg_param is not None) and (J0 is None):
            self.lambd = rots_reg_param * np.ones((self.N,))
            self.J = None
        else:
            assert J0 is not None
            self.lambd = None
            self.J = min(int(J0 * self.n ** ((2 - 3 * self.eta) / 5)), self.n - 1)

        self.data_discrepancy = np.zeros((self.n, self.N))  # (\|Ag.u - f_i\|^2)_g,i

        self.rots_batch_size = rots_batch_size

        self.max_iter = max_iter
        self.save_iterates = save_iterates

    @property
    def angles(self):
        return self._points.as_euler("ZYZ").astype(self.dtype)

    @angles.setter
    def angles(self, values):
        self._points = R.from_euler("ZYZ", values)

    @property
    def rots(self):
        return self._points.as_matrix().astype(self.dtype)

    @rots.setter
    def rots(self, values):
        self._points = R.from_matrix(values)

    @property
    def quaternions(self):
        quats = np.roll(self._points.as_quat().astype(self.dtype), 1, axis=-1)
        sign_s = np.sign(quats[:, 0])
        sign_s[sign_s == 0] = 1
        return quaternionic.array(sign_s[:, None] * quats).normalized.ndarray.astype(self.dtype)

    @quaternions.setter
    def quaternions(self, values):
        quats = quaternionic.array(np.roll(values, -1, axis=-1)).normalized.ndarray
        self._points = R.from_quat(quats)

    def get_cost(self):  # DEPR
        data_term = np.sum(1 / (2 * self.sigmas[None, :]) * self.data_discrepancy * self.rots_coeffs)
        vol_reg_term = 1 / (2 * self.tau) * np.sum(self.vol.asnumpy() ** 2)   # / self.L ** 3
        rot_reg_term = 1/ (2 * self.n ** self.eta) * np.sum(self.lambd[None] * self.rots_coeffs ** 2)

        cost = data_term + vol_reg_term + rot_reg_term
        return cost

    def get_solver_result(self):
        return self.vol

    def data_discrepancy_update(self):
        N = self.N
        n = self.n
        dtype = self.dtype

        logger.info("Computing \|Ag.u - f_i\|^2")
        im = self.images.asnumpy()
        F = np.zeros((n, N), dtype=dtype)
        F3 = np.sum(im ** 2, axis=(1, 2))[None, :]

        for start in range(0, n, self.rots_batch_size):
            all_idx = np.arange(start, min(start + self.rots_batch_size, n))

            rots_sampling_projections = self.forward(self.vol, self.integrator.rots[all_idx]).asnumpy()

            F1 = np.sum(rots_sampling_projections ** 2, axis=(1, 2))[:, None]
            F2 = - 2 * np.einsum("ijk,gjk->gi", im, rots_sampling_projections)
            F[all_idx, :] = (F1 + F2 + F3)  # / (L ** 2)  # 2 * self.plan.squared_noise_level missing now

            logger.info(
                "Computing data fidelity for {} rotations and {} images at {}%".format(n, N, int((all_idx[-1] + 1) / n * 100)))

        self.data_discrepancy = F

    def forward(self, vol, rots):
        """
        Apply forward image model to volume
        :param vol: A volume instance.
        :param start: Start index of image to consider
        :param num: Number of images to consider
        :return: The images obtained from volume by projecting, applying CTFs, translating, and multiplying by the
            amplitude.
        """
        im = vol.project(0, rots)
        im = self.eval_filter(im)  # Here we only use 1 filter, but might as well do one for every entry
        # im = im.shift(self.offsets[all_idx, :])
        im *= self.amplitude  # [im.n, np.newaxis, np.newaxis]  # Here we only use 1 amplitude,
        # but might as well do one for every entry

        return im

    def adjoint_forward(self, im, rots):
        """
        Apply adjoint mapping to set of images
        :param im: An Image instance to which we wish to apply the adjoint of the forward model.
        :param start: Start index of image to consider
        :return: An L-by-L-by-L volume containing the sum of the adjoint mappings applied to the start+num-1 images.
        """
        res = np.zeros((self.L, self.L, self.L), dtype=self.dtype)

        img = im.copy()
        img *= self.amplitude
        # im = im.shift(-self.offsets[all_idx, :])
        img = self.eval_filter(img)

        res += img.backproject(rots)[0]

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
