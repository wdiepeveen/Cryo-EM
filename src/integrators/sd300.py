import numpy as np
import os

from src.integrators import SO3_Integrator


class SD300(SO3_Integrator):
    """Hexacosichoron Integration"""

    def __init__(self, dtype=np.float32):

        # Read quaternions from text file
        data_dir = os.path.join(os.path.dirname(__file__), "..", "..", "data", "points")
        filename = "sdr011_00600.txt"
        filepath = os.path.join(data_dir, filename)

        all_quats = np.array_split(np.loadtxt(filepath, dtype=dtype), [4], axis=1)[0]

        # Remove SO3 duplicates
        reference_dir = np.array([1.0, 1e-4, 1.1e-4, 1.5e-4])
        quatskeep = (all_quats.dot(reference_dir) > 0)
        quaternions = all_quats[quatskeep]

        sep_dist = 0.54184  # See Womersly, Efficient Spherical Designs with Good Geometric Properties
        mesh_norm = 0.77628  # See Womersly, Efficient Spherical Designs with Good Geometric Properties
        tri_dist = 1.55256

        super().__init__(quaternions, sep_dist, mesh_norm, tri_dist, dtype=dtype)
