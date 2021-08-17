import numpy as np
import os

import spherical

from solvers.lifting.integration import Integrator


class HexacosichoronIntegrator(Integrator):

    def __init__(self,
                 ell_max=3,
                 dtype=np.float32,
                 ):
        super().__init__(dtype=dtype, n=300, ell_max=ell_max, t=5)

        # Read quaternions from text file
        data_dir = os.path.join(os.path.dirname(__file__), "points")
        filename = "sdr011_00600.txt"
        filepath = os.path.join(data_dir, filename)

        all_quats = np.array_split(np.loadtxt(filepath, dtype=self.dtype), [4], axis=1)[0]

        # Rearrange quaternions: scipy needs the last entry to be the scalar part
        # all_quats = np.roll(all_quats, -1, axis=-1)

        # Remove SO3 duplicates
        # reference_dir = np.array([1e-4, 1.1e-4, 1.5e-4, 1.0])
        reference_dir = np.array([1.0, 1e-4, 1.1e-4, 1.5e-4])
        quatskeep = (all_quats.dot(reference_dir) > 0)
        quaternions = all_quats[quatskeep]

        self.quaternions = quaternions
        self.initialize_manifold()
        self.initialize_b2w()
