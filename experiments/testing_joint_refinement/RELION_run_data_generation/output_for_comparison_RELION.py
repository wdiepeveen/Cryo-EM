import numpy as np
import mrcfile
import os
import pickle


from tools.exp_tools import Exp

# Directly start experiment
exp = Exp()

exp.begin(prefix="RELION_run")

results_dir = "results"
results_folder = "expC3_22-04-06_23-59-48_SNR16_N2048_J15_r2_s10_i10"  # spike protein
results_path = os.path.join("..", "..", "..", results_dir, results_folder)

solver_data = exp.open_pkl(results_path, "solver_data_r2")

solver = solver_data["solver"]
voxel_size = solver_data["voxel_size"]

images = solver.plan.images
exp.save_mrcs("sign_flipped_images", - images.asnumpy().astype(np.float32), voxel_size=voxel_size)


