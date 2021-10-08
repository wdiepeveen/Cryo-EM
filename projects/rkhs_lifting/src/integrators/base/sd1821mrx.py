import numpy as np
import os

from projects.rkhs_lifting.src.integrators.base import SO3_Integrator
from projects.rkhs_lifting.src.integrators.base.sd1821 import SD1821


class SD1821MRx(SO3_Integrator):
    """x times Refined Spherical Design Integration with base n=1821"""

    def __init__(self, repeat=1, dtype=np.float32):
        if repeat == 0:
            integrator = SD1821()
            super().__init__(integrator.quaternions, integrator.sep_dist, integrator.mesh_norm, integrator.tri_dist, dtype=dtype)
        elif repeat >= 1:
            # Read quaternions from text file
            data_dir = os.path.join(os.path.dirname(__file__), "..", "..", "..", "..", "..", "data", "points", "refined")
            filename = "sd1821mr{}.npy".format(int(repeat))
            path = os.path.join(data_dir, filename)

            quaternions = np.load(path)

            sep_dist = 0.29224 * 2**(-repeat)
            mesh_norm = 0.29242 * 2**(-repeat)
            tri_dist = 0.57060 * 2**(-repeat)

            super().__init__(quaternions, sep_dist, mesh_norm, tri_dist, dtype=dtype)
