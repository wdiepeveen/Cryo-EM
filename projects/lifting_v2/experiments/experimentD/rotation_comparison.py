import numpy as np
import mrcfile
import os
import pickle
import matplotlib.pyplot as plt

from aspire.utils.coor_trans import (
    get_aligned_rotations,
    get_rots_mse,
    register_rotations,
)

from tools.exp_tools import Exp
from projects.lifting_v2.src.util.rots_container import RotsContainer
from scipy.spatial.transform import Rotation as R

# Directly start experiment
exp = Exp()

exp.begin(prefix="RELION_comparison")

results_dir = "results"
results_folder = "expC3_22-04-06_23-59-48_SNR16_N2048_J15_r2_s10_i10"  # spike protein
results_path = os.path.join("..", "..", "..", "..", results_dir, results_folder)

solver_data = exp.open_pkl(results_path, "solver_data_r2")

solver = solver_data["solver"]
rots_gt = solver_data["rots_gt"]
num_imgs = solver.plan.N

RELION_dir = "RELION_run"
RELION_folder = "RELION_run_rotations"
RELION_path = os.path.join(RELION_dir, RELION_folder)

RELION_rots_, RELION_trans_ = exp.open_pkl(RELION_path, "Large_set_run_it050_data_pose")
rots_RELION = RotsContainer(num_imgs, dtype=solver.plan.dtype)
# We have to transpose as we are using g rather than g^-1 in RELION
rots_RELION.rots = RELION_rots_.transpose(0,2,1)


# save rots in container
# Get register rotations after performing global alignment
Q_mat, flag = register_rotations(rots_RELION.rots, rots_gt.rots)
regrot = RotsContainer(num_imgs, dtype=solver.plan.dtype)
regrot.rots = get_aligned_rotations(rots_RELION.rots, Q_mat, flag)
mse_reg_est = get_rots_mse(regrot.rots, rots_gt.rots)
print(regrot.rots[0])
print(
    f"MSE deviation of the RELION estimated rotations using register_rotations : {mse_reg_est}"
)

# dist_est = solver.plan.integrator.manifold.dist(rots_RELION.quaternions[:, None, :],
#                                                 rots_gt.quaternions[:, None, :]).squeeze()
dist_est = solver.plan.integrator.manifold.dist(regrot.quaternions[:, None, :],
                                                rots_gt.quaternions[:, None, :]).squeeze()

print(dist_est)

plt.figure()
plt.hist(180 / np.pi * dist_est, bins=100)
plt.show()
