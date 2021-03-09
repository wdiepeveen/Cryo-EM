import logging
import numpy as np

from scipy.linalg import norm

from aspire.source.simulation import Simulation
from aspire.volume import Volume

# Own functions
from solvers.primaldual import l2_mse_data_fidelity, l2_grad_norm, so3_distance
from solvers.primaldual import differential_forward_rot,adjoint_differential_forward_rot
from solvers.primaldual import gradient_l2_mse_data_fidelity, gradient_l2_grad_norm
from solvers.primaldual import prox_dual_l2_mse_data_fidelity, prox_constraint_so3_distance
from noise.noise import SnrNoiseAdder
from solvers.primaldual import PrimalDualProblem
from solvers.primaldual import PrimalDualSolver


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
    vol_init = sim_data["vol_init"]  # Volume 65L
    rots_init = sim_data["rots_init"]  # np.Array
    old_sim = sim_data["sim"]  # Simulation

    vol_gt = sim_data["vol_gt"] # Volume 65L
    rots_gt = sim_data["rots_gt"] # np.Array

    # Define a precision for this experiment
    dtype = old_sim.dtype

    # Generate data at experiment resolution
    sim = Simulation(L=vol_gt.shape[1], n=old_sim.n, vols=vol_gt,
                     unique_filters=old_sim.unique_filters, dtype=dtype)
    sim.noise_adder = SnrNoiseAdder(seed=sim.seed, snr=snr)
    sim.rots = rots_gt  # Use true old rotations

    # Estimate simgma
    sigma = 1 / (1 + snr) * np.mean(np.var(sim.images(0, np.inf).asnumpy(), axis=(1, 2)))
    print("sigma = {}".format(sigma))

    prob = PrimalDualProblem(vols=vol_init,
                             data=sim.images(0, np.inf),
                             # rots=rots_gt,  # TODO remove this and use rots_init for real exp
                             rots=rots_init,
                             unique_filters=sim.unique_filters,
                             offsets=sim.offsets,  # we don't know the offsets at this point
                             noise_var=sigma
                             )

    mu1 = 1e9 / sim.L ** 2  # 0.31 / sim.L ** 2
    mu2 = 1.

    def cost_function(src):
        lam1 = mu1 * src.noise_var / src.n
        lam2 = mu2 * src.noise_var / src.n

        fidelity_penalty = l2_mse_data_fidelity(src)
        gradient_penalty = lam1 * l2_grad_norm(src)
        rotations_penalty = lam2 * so3_distance(src)
        cost = fidelity_penalty + gradient_penalty + rotations_penalty
        logger.info("data fidelity penalty = {} | gradient penalty = {} | rotations penalty = {}".format(fidelity_penalty,
                                                                                                         gradient_penalty,
                                                                                                         rotations_penalty)
                    )
        return cost * 1e10  # scaled cost

    def gradient_u(src):
        lam1 = mu1 * src.noise_var / src.n

        fidelity_gradient = gradient_l2_mse_data_fidelity(src)
        gradient_gradient = Volume(lam1 * gradient_l2_grad_norm(src).asnumpy()[0])
        gradient = fidelity_gradient + gradient_gradient
        logger.info(
            "data fidelity gradient = {} | gradient gradient = {}".format(norm(fidelity_gradient.asnumpy()[0]), norm(gradient_gradient.asnumpy()[0]))
            )
        return gradient

    def prox_primal_g(src, a, x):
        lam2 = mu2 * src.noise_var / src.n
        # TODO print a*lam2

        # prox = prox_so3_distance(src, a*lam2, x)
        prox = prox_constraint_so3_distance(src, a * lam2, x, radius=np.pi/2)
        logger.info(
            "primal distance = {}".format(a * lam2)
        )
        return prox

    def prox_dual_g(src, a, x):

        prox = prox_dual_l2_mse_data_fidelity(src, a, x)
        logger.info(
            "dual distance = {}".format(a)
        )
        return prox

    solver = PrimalDualSolver(problem=prob, cost=cost_function,
                              max_it=max_it,
                              gradient_u=gradient_u,
                              prox_primal_g=prox_primal_g,
                              prox_dual_g=prox_dual_g,
                              differential_g=differential_forward_rot,
                              adjoint_g=adjoint_differential_forward_rot,
                              sigma=1/2*1e12,
                              tau=1/2*1e-2,
                              gamma=0.2*1e-10,
                              )

    solver.solve(return_result=False)

    # TODO solve normal equations for prestop (update sim and estimate())

    # Save result
    exp.save("solver_data_{}snr_{}n".format(int(1 / snr), num_imgs),
             # Data
             ("sim", sim),
             ("vol_init", vol_init),  # (img_size,)*3
             ("rots_init", rots_init),
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
