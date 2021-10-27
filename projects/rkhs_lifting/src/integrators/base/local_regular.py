import numpy as np
import quaternionic

from projects.rkhs_lifting.src.integrators.base import SO3_Integrator
from projects.rkhs_lifting.src.manifolds.so3 import SO3


class Local_Regular(SO3_Integrator):
    """Local regular "cubic" grid Integration"""

    def __init__(self, quaternion=None, l=3, sep_dist=np.pi / 180, dtype=np.float32):
        assert l > 1
        so3 = SO3()
        # Discretize tangent space
        grid1D = sep_dist * np.linspace(-1., 1., l)
        x, y, z = np.meshgrid(grid1D, grid1D, grid1D)

        tvectors = np.zeros((l, l, l, 4))
        tvectors[:, :, :, 1] = x
        tvectors[:, :, :, 2] = y
        tvectors[:, :, :, 3] = z

        # Flatten tvectors
        v = tvectors.reshape((l ** 3, 4))

        e = np.zeros(v.shape)
        e[:, 0] = 1.  # Unit element

        quaternion_grid = so3.exp(e, v)  # Grid in TeSO(3)

        # Transport mesh over manifold through group action
        if quaternion is None:
            quaternion = e[0]

        g1 = quaternionic.array(quaternion[None]).normalized
        g2 = quaternionic.array(quaternion_grid).normalized
        quaternion_grid = (g1 * g2).ndarray

        # print(quaternion_grid.shape)

        super().__init__(quaternion_grid, sep_dist, 1 / np.sqrt(2) * sep_dist, np.sqrt(2) * sep_dist, dtype=dtype)
        self.l = l

    def update(self, quaternion=None):
        assert quaternion is not None
        return Local_Regular(quaternion=quaternion, l=self.l, sep_dist=self.sep_dist, dtype=self.dtype)
