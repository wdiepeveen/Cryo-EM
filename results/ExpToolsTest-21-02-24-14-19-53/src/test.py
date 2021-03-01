import logging
import os

import mrcfile
import numpy as np

from aspire.volume import Volume


# Own functions
from tools.exp_tools import exp_open
from functions.cost_functions import l2_data_fidelity, l2_grad_norm, so3_distance
from functions.differentials import differential_forward_rot,adjoint_differential_forward_rot
from functions.gradients import gradient_l2_data_fidelity, gradient_l2_grad_norm
from functions.proximal_maps import prox_so3_distance, prox_dual_l2_data_fidelity
from problems.primal_dual_problem import PrimalDualProblem
from solvers.primal_dual_solver import PrimalDualSolver

logger = logging.getLogger(__name__)


# TODO test whether cost_functions work
DATA_DIR = os.path.join(os.path.dirname(__file__), "results/")
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results/")

logger.info(
    "This script illustrates orientation refinement using "
    "lRCPA as primal-dual splitting method"
)

# Define a precision for this experiment
dtype = np.float32

# Load data
infile = mrcfile.open(os.path.join(DATA_DIR, "reconstructed70SRibosome_vol_65p.mrc"))
vol = Volume(infile.data.astype(dtype)) # Load initial volume
sim = exp_open(os.path.join(DATA_DIR, "sim_up.pkl")) # Load simulator

prob = PrimalDualProblem(vols=vol, data=sim.images(0,np.inf), rots=sim.rots, unique_filters=sim.unique_filters)

lam1 = 1e-4
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
                          max_it=5,
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

with mrcfile.new(os.path.join(RESULTS_DIR, "joint_reconstructed70SRibosome_vol_{}p.mrc".format(prob.L)), overwrite=True) as mrc:
    mrc.set_data(volume)

# Test cost_funtions
# logger.info(
#     "Testing data fidelity"
# )
# print(l2_data_fidelity(prob))
#
# logger.info(
#     "Testing grad norm fidelity"
# )
# print(l2_grad_norm(prob))
#
# logger.info(
#     "Testing rotation fidelity"
# )
#
# # determinant 1.
#
# print(so3_distance(prob))
#
# logger.info(
#     "Testing differential"
# )
# eta = np.arange(9).reshape((3, 3))
# eta = -1/2*(eta - eta.T)
#
# xi = differential_forward_rot(prob, eta)
# print(xi)
#
# logger.info(
#     "Testing adjoint differential"
# )
#
# print(adjoint_differential_forward_rot(prob,xi))
#
# logger.info(
#     "Testing gradient"
# )
#
# print(gradient_l2_data_fidelity(prob))