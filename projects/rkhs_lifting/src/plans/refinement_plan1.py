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
                 vol=None,
                 rots=None,
                 stop=None,  # TODO here a default stopping criterion
                 stop_rots_gd=None,  # TODO here a default stopping criterion
                 squared_noise_level=None,
                 images=None,
                 filter=None,
                 amplitude=None,
                 kernel=None,
                 integrator=None,
                 rots_batch_size=8192,
                 dtype=np.float32,
                 seed=0,
                 ):

        if images is None:
            raise RuntimeError("No data provided")
        else:
            assert isinstance(images, Image)

        self.p = Refinement_Problem1(images=images,
                                     filter=filter,
                                     amplitude=amplitude,
                                     kernel=kernel,
                                     integrator=integrator,
                                     dtype=dtype,
                                     seed=seed)

        self.o = Refinement_Options1(squared_noise_level=squared_noise_level,
                                     stop=stop,
                                     stop_rots_gd=stop_rots_gd,
                                     rots_batch_size=rots_batch_size,
                                     )

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

        self.vol = vol
        self._points = None
        self.rots = rots

    # TODO check the properties and the setters for bugs

    @property
    def angles(self):
        return self._points.as_euler("ZYZ").astype(self.p.dtype)

    @angles.setter
    def angles(self, values):
        self._points = R.from_euler("ZYZ", values)

    @property
    def rots(self):
        return self._points.as_matrix().astype(self.p.dtype)

    @rots.setter
    def rots(self, values):
        self._points = R.from_matrix(values)

    @property
    def quaternions(self):
        quats = np.roll(self._points.as_quat().astype(self.p.dtype), 1, axis=-1)
        sign_s = np.sign(quats[:, 0])
        sign_s[sign_s == 0] = 1
        return quaternionic.array(sign_s[:, None] * quats).normalized.ndarray.astype(self.p.dtype)

    @quaternions.setter
    def quaternions(self, values):
        quats = quaternionic.array(np.roll(values, -1, axis=-1)).normalized.ndarray
        self._points = R.from_quat(quats)

    # def quaternion_to_rot(self, quat):
    #     quats = quaternionic.array(np.roll(quat, -1, axis=-1)).normalized.ndarray
    #     tmp = R.from_quat(quats)
    #     return tmp.as_matrix().astype(self.p.dtype)

    def get_solver_result(self):
        return self.vol

    def get_cost(self, index=None, quaternion=None):
        assert index is not None

        if quaternion is None:
            quaternion = self.quaternions[index,:]

        im = self.p.images.asnumpy()[index,:,:]
        # qs = np.zeros((self.p.n,), dtype=self.p.dtype)
        integrator = self.p.integrator.update(quaternions=quaternion)

        rots_sampling_projections = self.forward(self.vol, integrator.rots).asnumpy()  # TODO check whether we get correct rot input here

        q1 = np.sum(rots_sampling_projections ** 2, axis=(1, 2))
        q2 = - 2 * np.einsum("jk,gjk->g", im, rots_sampling_projections)
        q3 = np.sum(im ** 2)

        integrands = ((q1 + q2 + q3) / (2 * self.o.squared_noise_level * self.p.L ** 2)).astype(self.p.dtype)

        cost = integrator.integrate(integrands)  # TODO we need the kernel for this

        return cost

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
        im *= self.p.amplitude  # [im.n, np.newaxis, np.newaxis]  # Here we only use 1 amplitude,
        # but might as well do one for every entry

        return im

    def adjoint_forward(self, im, rots):
        """
        Apply adjoint mapping to set of images
        :param im: An Image instance to which we wish to apply the adjoint of the forward model.
        :param start: Start index of image to consider
        :return: An L-by-L-by-L volume containing the sum of the adjoint mappings applied to the start+num-1 images.
        """
        res = np.zeros((self.p.L, self.p.L, self.p.L), dtype=self.p.dtype)
        for start in range(0, self.p.N, self.o.rots_batch_size):
            logger.info(
                "Running through projections {}/{} = {}%".format(start, self.p.N, np.round(start / self.p.N * 100, 2)))
            all_idx = np.arange(start, min(start + self.o.rots_batch_size, self.p.N))

            img = im[all_idx, :, :]
            img *= self.p.amplitude
            # im = im.shift(-self.offsets[all_idx, :])
            img = self.eval_filter(img)

            res += img.backproject(rots)[0]

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
