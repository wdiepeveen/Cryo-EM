import numpy as np
import os
import logging

from src.integrators.sd60 import SD60
from src.mesh.midpoint_refinement import Midpoint_Refinement

logger = logging.getLogger(__name__)

repeat = 5

# Output location quaternions text file
data_dir = os.path.join(os.path.dirname(__file__), "..", "..", "..", "data", "points", "refined")
filename = "sd60mr{}.npy".format(int(repeat))
path = os.path.join(data_dir, filename)

# Check if file exists (assert does not exist)

# Load integrator
base_integrator = SD60()
resolution = base_integrator.tri_dist * 2 ** (-repeat)

# Do meshing
refiner = Midpoint_Refinement(quats=base_integrator.quaternions, h=resolution)

# Save file as .npy
with open(path, "wb") as f:
    np.save(path, refiner.verts)
