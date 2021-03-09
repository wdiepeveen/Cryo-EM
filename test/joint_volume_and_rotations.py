import logging
import matplotlib.pyplot as plt

import numpy as np

from scipy.ndimage import zoom

from aspire.volume import Volume
from aspire.source.simulation import Simulation


# Own functions
from solvers.primaldual import l2_data_fidelity, l2_grad_norm, so3_distance
from solvers.primaldual import differential_forward_rot,adjoint_differential_forward_rot
from solvers.primaldual import gradient_l2_data_fidelity, gradient_l2_grad_norm
from solvers.primaldual import prox_so3_distance, prox_dual_l2_data_fidelity
from solvers.primaldual import PrimalDualProblem
from solvers.primaldual import PrimalDualSolver
from tools.exp_tools import Exp


logger = logging.getLogger(__name__)

exp = Exp()
exp.begin(prefix="joint_reconstruction",postfix="65L_128n")

logger.info(
    "This script illustrates orientation refinement using "
    "lRCPA as primal-dual splitting method"
)

# Define a precision for this experiment
dtype = np.float32

# Load data
DATA_DIR = "results/data_simulation_21-02-27_14-37-10_33p_128n"

sim_data = exp.open_pkl(DATA_DIR, "simulation_data")
vol_init = sim_data["vol_init"]  # Volume 33L
rot_init = sim_data["rot_init"]  # np.Array
old_sim = sim_data["sim"]  # Simulation

vol_gt = sim_data["vol_gt"] # Volume 65L
rots_gt = sim_data["rots_gt"] # np.Array

# Generate data at resolution 65-by-65-by-65
sim = Simulation(L=vol_gt.shape[1], n=old_sim.n, vols=vol_gt,
                 unique_filters=old_sim.unique_filters, dtype=dtype)
sim.rots = rots_gt  # Use old rotations

# Construct problem
zoomed_vol_init = Volume(zoom(vol_init.asnumpy()[0], 65./33, order=1))
# Save zoomed_init_volume to results_folder
exp.save_mrc("initialization_65p_128n", zoomed_vol_init.asnumpy()[0])
exp.save_mrc("ground_truth_65p", vol_gt.asnumpy()[0])

prob = PrimalDualProblem(vols=zoomed_vol_init,
                         data=sim.images(0,np.inf),
                         rots=sim.rots,
                         unique_filters=sim.unique_filters,
                         offsets=sim.offsets  # we don't know the offsets at this point
                         )

lam1 = 0.1/sim.L**2
lam2 = 1.

def cost_function(src):
    cost = l2_data_fidelity(src) + lam1*l2_grad_norm(src) + lam2*so3_distance(src)
    return cost

def gradient_u(src):
    gradient = gradient_l2_data_fidelity(src) + lam1*gradient_l2_grad_norm(src)
    return gradient

def prox_primal_g(src, a, x):
    prox = prox_so3_distance(src, a*lam2, x)
    return prox


solver = PrimalDualSolver(problem=prob, cost=cost_function,
                          max_it=2,
                          gradient_u=gradient_u,
                          prox_primal_g=prox_primal_g,
                          prox_dual_g=prox_dual_l2_data_fidelity,
                          differential_g=differential_forward_rot,
                          adjoint_g=adjoint_differential_forward_rot
                          )

# TODO test whether we won't get any change in cost function once our resolution L grows
# TODO test whether for lam1=0. we get our old volume back
volume = solver.solve()
cost = solver.cost
relerror_u = solver.relerror_u
relerror_g = solver.relerror_g
relerror_tot = solver.relerror_tot

# check error on the rotations
# TODO distance error with init
# TODO total rotation MSE

# Process output
num_its = len(cost)

relerror_fig = plt.figure()

plt.figure()
plt.plot(np.arange(num_its)+1, cost)
exp.savefig("costfig")
# TODO new plt?

# plt.figure()
# plt.plot(np.arange(num_its)+1, relerror_u)
# exp.savefig("relerror_u_fig")
relerror_fig.plot(np.arange(num_its)+1, relerror_u)
exp.savefig("relerror_u_fig")

plt.figure()
plt.plot(np.arange(num_its)+1, relerror_g)
exp.savefig("relerror_g_fig")

plt.figure()
plt.plot(np.arange(num_its)+1, relerror_tot)
exp.savefig("relerror_fig")

# Save results
exp.save_mrc("result_65p_128n", volume.asnumpy()[0])





