import numpy as np
import logging

import quaternionic

from scipy.spatial.transform import Rotation as R

from aspire.image import Image
from aspire.utils.coor_trans import grid_2d
from aspire.volume import Volume

from projects.rkhs_lifting.src.plans import Plan
from projects.rkhs_lifting.src.plans.problems.refinement_problem1 import Refinement_Problem1
from projects.rkhs_lifting.src.plans.options.refinement_options1 import Refinement_Options1

logger = logging.getLogger(__name__)


class Refinement_Plan1(Plan):
    """Class for preprocessing inputs and defining several functions such as cost and forward operator"""

    def __init__(self,
                 quaternions=None,
                 rots_coeffs=None,
                 squared_noise_level=None,
                 images=None,
                 filter=None,
                 amplitude=None,
                 volume_reg_param=None,
                 rots_batch_size=8192,
                 dtype=np.float32,
                 seed=0,
                 ):

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

        self.lam1 = volume_reg_param

        self.vol = None
        self._points = None
        self.quaternions = quaternions

        self.rots_coeffs = rots_coeffs

        self.rots_batch_size = rots_batch_size

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

    def get_solver_result(self):
        return self.vol

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
        res = np.zeros((self.L, self.L, self.L), dtype=self.dtype)
        for start in range(0, self.N, self.rots_batch_size):
            logger.info(
                "Running through projections {}/{} = {}%".format(start, self.N, np.round(start / self.N * 100, 2)))
            all_idx = np.arange(start, min(start + self.rots_batch_size, self.N))

            img = Image(im[all_idx, :, :])
            img *= self.amplitude
            # im = im.shift(-self.offsets[all_idx, :])
            img = self.eval_filter(img)

            res += img.backproject(self.rots[all_idx])[0]

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
