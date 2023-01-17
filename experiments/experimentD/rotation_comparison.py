import numpy as np
import os
import matplotlib.pyplot as plt

from aspire.utils.coor_trans import (
    get_aligned_rotations,
    get_rots_mse,
    register_rotations,
)

from tools.exp_tools import Exp
from src.util import RotsContainer

plt.rcParams.update({'font.size': 18})
plt.rcParams.update({'figure.autolayout': True})

# Directly start experiment
exp = Exp()

exp.begin(prefix="RELION_comparison")

results_dir = "results"
results_folder = "expC3_23-01-07_09-47-03_SNR16_N2048_J15_r2_s10_i10"  # spike protein
results_path = os.path.join("..", "..", "..", "..", results_dir, results_folder)

solver_data = exp.open_pkl(results_path, "solver_data_r2")

solver = solver_data["solver"]
voxel_size = solver_data["voxel_size"]
rots_gt = solver_data["rots_gt"]
num_imgs = solver.plan.N

RELION_dir = "../testing_joint_refinement/RELION_run"
RELION_folder = "RELION_run_rotations"
RELION_path = os.path.join(RELION_dir, RELION_folder)

# RELION_rots_, RELION_trans_ = exp.open_pkl(RELION_path, "Large_set_run_it050_data_pose")
RELION_rots_, RELION_trans_ = exp.open_pkl(RELION_path, "run_data")
rots_RELION = RotsContainer(num_imgs, dtype=solver.plan.dtype)
# We have to transpose as we are using g rather than g^-1 in RELION
trans_mat = np.array([[1,0,0], [0,0,1], [0,1,0]])
rots_RELION.rots = np.einsum("ik,Nkl,jl->Nij",trans_mat, RELION_rots_, trans_mat)
# rots_RELION.rots = np.einsum("ik,Nkl,jl->Nij",trans_mat, RELION_rots_.transpose(0,2,1), trans_mat)
# rots_RELION.rots = RELION_rots_.transpose(0,2,1)
# rots_RELION.rots = RELION_rots_


# save rots in container
# Get register rotations after performing global alignment
Q_mat, flag = register_rotations(rots_RELION.rots, rots_gt.rots)
regrot = RotsContainer(num_imgs, dtype=solver.plan.dtype)
regrot.rots = get_aligned_rotations(rots_RELION.rots, Q_mat, flag)
mse_reg_est = get_rots_mse(regrot.rots, rots_gt.rots)
i = 1
print("RELION rot =\n {} \n reg RELION rot =\n {} \n GT rot =\n {}".format(rots_RELION.rots[i], regrot.rots[i], rots_gt.rots[i]))
print("RELION angles = {} | reg RELION angles = {}| GT angles = {}".format(rots_RELION.angles[i],regrot.angles[i], rots_gt.angles[i]))
print(
    f"MSE deviation of the RELION estimated rotations using register_rotations : {mse_reg_est}"
)

non_registered_dist_est = solver.plan.integrator.manifold.dist(rots_RELION.quaternions[:, None, :],
                                                rots_gt.quaternions[:, None, :]).squeeze()
dist_est = solver.plan.integrator.manifold.dist(regrot.quaternions[:, None, :],
                                                rots_gt.quaternions[:, None, :]).squeeze()
print(non_registered_dist_est)
print(dist_est)

# plt.figure()
# plt.hist(180 / np.pi * non_registered_dist_est, bins=100)
# plt.xlabel(r"Error $(\degree)$")
# plt.ylabel("Frequency")
# plt.show()

plt.figure()
plt.hist(180 / np.pi * dist_est, bins=100)
plt.xlabel(r"Error $(\degree)$")
plt.ylabel("Frequency")
plt.autoscale()
plt.xlim([0,180])
plt.ylim([0, 400])
exp.save_fig("RELION_manifold_error", save_eps=True)
plt.show()

# Get register rotations after performing global alignment
Q_mat, flag = register_rotations(solver.rots_iterates[-1], rots_gt.rots)
regrot = RotsContainer(num_imgs, dtype=solver.plan.dtype)
regrot.rots = get_aligned_rotations(solver.rots_iterates[-1], Q_mat, flag)
mse_reg_est = get_rots_mse(regrot.rots, rots_gt.rots)

dist_est = solver.plan.integrator.manifold.dist(regrot.quaternions[:, None, :],
                                                rots_gt.quaternions[:, None, :]).squeeze()
# print("dist_est.shape = {}".format(dist_est.shape))
plt.figure()
plt.hist(180 / np.pi * dist_est, bins=100)
plt.xlabel(r"Error $(\degree)$")
plt.ylabel("Frequency")
plt.autoscale()
plt.xlim([0,180])
plt.ylim(0, 400)
exp.save_fig("distance_est_ESL", save_eps=True)
plt.show()

# angle plots
angle_error = rots_gt.angles - rots_RELION.angles
print(rots_gt.angles)
# print(angle_error)
# alpha
plt.figure()
plt.hist((((angle_error[:,0] + np.pi) % (2 * np.pi)) - np.pi) / np.pi * 180, bins=100)
plt.xlabel(r"Error $(\degree)$")
plt.ylabel("Frequency")
plt.autoscale()
plt.xlim([-180,180])
plt.ylim([0, 400])
exp.save_fig("RELION_alpha_error", save_eps=True)
plt.show()
# beta
plt.figure()
plt.hist((((angle_error[:,1] + np.pi/2) % np.pi) - np.pi/2) / np.pi * 180, bins=100)
plt.xlabel(r"Error $(\degree)$")
plt.ylabel("Frequency")
plt.autoscale()
plt.xlim([-90,90])
plt.ylim([0, 400])
exp.save_fig("RELION_beta_error", save_eps=True)
plt.show()
# gamma
plt.figure()
plt.hist((((angle_error[:,2] + np.pi) % (2 * np.pi)) - np.pi) / np.pi * 180, bins=100)
plt.xlabel(r"Error $(\degree)$")
plt.ylabel("Frequency")
plt.autoscale()
plt.xlim([-180,180])
plt.ylim([0, 400])
exp.save_fig("RELION_gamma_error", save_eps=True)
plt.show()

# angle plots
ESL_angle_error = rots_gt.angles - solver.plan.angles
# print(rots_gt.angles)
# print(angle_error)
# alpha
plt.figure()
plt.hist((((ESL_angle_error[:,0] + np.pi) % (2 * np.pi)) - np.pi) / np.pi * 180, bins=100)
plt.xlabel(r"Error $(\degree)$")
plt.ylabel("Frequency")
plt.autoscale()
plt.xlim([-180,180])
plt.ylim([0, 400])
exp.save_fig("ESL_alpha_error", save_eps=True)
plt.show()
# beta
plt.figure()
plt.hist((((ESL_angle_error[:,1] + np.pi/2) % np.pi) - np.pi/2) / np.pi * 180, bins=100)
plt.xlabel(r"Error $(\degree)$")
plt.ylabel("Frequency")
plt.autoscale()
plt.xlim([-90,90])
plt.ylim([0, 400])
exp.save_fig("ESL_beta_error", save_eps=True)
plt.show()
# gamma
plt.figure()
plt.hist((((ESL_angle_error[:,2] + np.pi) % (2 * np.pi)) - np.pi) / np.pi * 180, bins=100)
plt.xlabel(r"Error $(\degree)$")
plt.ylabel("Frequency")
plt.autoscale()
plt.xlim([-180,180])
plt.ylim([0, 400])
exp.save_fig("ESL_gamma_error", save_eps=True)
plt.show()

# set solver rots and do volume update step
# solver.plan.rots = rots_RELION.rots
# solver.volume_step()

# exp.save_mrc("result_vol", solver.plan.vol.asnumpy()[0].astype(np.float32),
#                      voxel_size=voxel_size)
