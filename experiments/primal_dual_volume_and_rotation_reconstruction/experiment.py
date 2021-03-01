import logging
import numpy as np

from aspire.source.simulation import Simulation

# Own functions
from functions.cost_functions import l2_data_fidelity, l2_grad_norm, so3_distance
from functions.differentials import differential_forward_rot,adjoint_differential_forward_rot
from functions.gradients import gradient_l2_data_fidelity, gradient_l2_grad_norm
from functions.proximal_maps import prox_so3_distance, prox_dual_l2_data_fidelity
from problems.primal_dual_problem import PrimalDualProblem
from solvers.primal_dual_solver import PrimalDualSolver


logger = logging.getLogger(__name__)


def experiment(exp=None,
                  num_imgs=None,
                  snr=1.,
                  max_it=20,
                  results_folder=None
                  ):

    logger.info(
        "This experiment illustrates orientation refinement using "
        "lRCPA as primal-dual splitting method"
    )

    # Load data
    if results_folder is not None:
        data_dir = results_folder
    else:
        data_dir = exp.results_folder

    sim_data = exp.open_pkl(data_dir, "simulation_data_{}snr_{}n".format(int(1/snr), num_imgs))
    vol_init = sim_data["vol_init"]  # Volume 33L
    rot_init = sim_data["rot_init"]  # np.Array
    old_sim = sim_data["sim"]  # Simulation

    vol_gt = sim_data["vol_gt"] # Volume 65L
    rots_gt = sim_data["rots_gt"] # np.Array

    # Define a precision for this experiment
    dtype = old_sim.dtype

    # Generate data at experiment resolution
    sim = Simulation(L=vol_gt.shape[1], n=old_sim.n, vols=vol_gt,
                     unique_filters=old_sim.unique_filters, dtype=dtype)
    sim.rots = rots_gt  # Use true old rotations

    sim1 = Simulation(L=vol_gt.shape[1], n=old_sim.n, vols=vol_gt,
                     unique_filters=old_sim.unique_filters, dtype=dtype, seed=31)

    prob = PrimalDualProblem(vols=vol_init,
                             data=sim.images(0,np.inf),
                             rots=sim1.rots,
                             rots_prior=rot_init,
                             # rots=rot_init,
                             unique_filters=sim.unique_filters,
                             offsets=sim.offsets  # we don't know the offsets at this point
                             )

    lam1 = 0.1 / sim.L**2
    lam2 = 1e-9

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
                              max_it=max_it,
                              gradient_u=gradient_u,
                              prox_primal_g=prox_primal_g,
                              prox_dual_g=prox_dual_l2_data_fidelity,
                              differential_g=differential_forward_rot,
                              adjoint_g=adjoint_differential_forward_rot
                              )

    solver.solve(return_result=False)

    # Save result
    exp.save("solver_data_{}snr_{}n".format(int(1 / snr), num_imgs),
             # Data
             ("sim", sim),
             ("vol_init", vol_init),  # (img_size,)*3
             ("rot_init", rot_init),
             ("vol_gt", vol_gt),  # (img_size,)*3
             ("rots_gt", rots_gt),
             # Results
             ("volume_est", solver.problem.vols),
             ("rots_est", solver.problem.rots),
             ("cost", solver.cost),
             ("relerror_u", solver.relerror_u),
             ("relerror_g", solver.relerror_g),
             ("relerror_tot", solver.relerror_tot)
             )