import numpy as np
import os

from projects.rkhs_lifting.src.integrators.base import SO3_Integrator


class SD60MRx(SO3_Integrator):
    """x times Refined Icosahedron Integration"""

    def __init__(self, repeat=1, dtype=np.float32):
        assert repeat >= 1

        # Read quaternions from text file
        data_dir = os.path.join(os.path.dirname(__file__), "..", "..", "..", "..", "..", "data", "points", "refined")
        filename = "sd60mr{}.npy".format(int(repeat))
        path = os.path.join(data_dir, filename)

        quaternions = np.load(path)

        sep_dist = 1.25664 * 2**(-repeat)
        mesh_norm = 0.77628 * 2**(-repeat)
        tri_dist = 1.25664 * 2**(-repeat)

        super().__init__(quaternions, sep_dist, mesh_norm, tri_dist, dtype=dtype)
