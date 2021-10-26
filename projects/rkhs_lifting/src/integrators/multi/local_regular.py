import numpy as np
import quaternionic

from projects.rkhs_lifting.src.integrators.multi import SO3_Multi_Integrator
from projects.rkhs_lifting.src.manifolds.so3 import SO3


class Local_Regular(SO3_Multi_Integrator):
    """Local regular "cubic" grid Integration"""

    def __init__(self, quaternions=None, l=3, sep_dist=np.pi / 180, dtype=np.float32):
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
        g1 = quaternionic.array(quaternions[None, :, :]).normalized
        g2 = quaternionic.array(quaternion_grid[:, None, :]).normalized
        quaternion_grid = (g1 * g2).ndarray

        sep_dists = sep_dist * np.ones((quaternion_grid.shape[0],))

        super().__init__(quaternions=quaternion_grid, sep_dists=sep_dists, mesh_norms=1 / np.sqrt(2) * sep_dists,
                         tri_dists=np.sqrt(2) * sep_dists, dtype=dtype)
        self.l = l

    def update(self, quaternions=None):
        assert quaternions is not None
        return Local_Regular(quaternions=quaternions, l=self.l, sep_dist=self.sep_dists[0], dtype=self.dtype)
