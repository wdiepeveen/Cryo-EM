import numpy as np
import os

from projects.rkhs_lifting.src.integrators.base import SO3_Integrator


class SD1821(SO3_Integrator):
    """Spherical Design Integration with n=1821"""

    def __init__(self, dtype=np.float32):

        # Read quaternions from text file
        data_dir = os.path.join(os.path.dirname(__file__), "points")
        filename = "sdr011_03642.txt"
        filepath = os.path.join(data_dir, filename)

        all_quats = np.array_split(np.loadtxt(filepath, dtype=self.dtype), [4], axis=1)[0]

        # Remove SO3 duplicates
        reference_dir = np.array([1.0, 1e-4, 1.1e-4, 1.5e-4])
        quatskeep = (all_quats.dot(reference_dir) > 0)
        quaternions = all_quats[quatskeep]

        super().__init__(quaternions, dtype=dtype)
