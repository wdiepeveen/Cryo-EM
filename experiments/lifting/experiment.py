import logging
import numpy as np

from aspire.basis import FBBasis3D
from aspire.source.simulation import Simulation

# Own functions
from solvers.lifting.functions.cost_functions import l2_data_fidelity, l2_vol_prior, l2_dens_prior, integral_dens_prior
from noise.noise import SnrNoiseAdder
from solvers.lifting.integration.hexacosichoron import HexacosichoronIntegrator
from solvers.lifting.update.density.qo import quadratic_optimisation_update
from solvers.lifting.update.volume.cg import conjugate_gradient_update
from solvers.lifting.problems.inside_norm import InsideNormLiftingProblem
from solvers.lifting.solver import LiftingSolver

logger = logging.getLogger(__name__)


def experiment(exp=None,
               num_imgs=None,
               snr=1.,
               max_it=20,
               results_folder=None
               ):
    logger.info(
        "This experiment illustrates orientation refinement using "
        "a lifting approach"
    )

    # Load data
    if results_folder is not None:
        data_dir = results_folder
    else:
        data_dir = exp.results_folder

    sim_data = exp.open_pkl(data_dir, "simulation_data_{}snr_{}n".format(int(1 / snr), num_imgs))
    vol_init = sim_data["vol_init"]  # Volume 65L
    rots_init = sim_data["rots_init"]  # np.Array
    old_sim = sim_data["sim"]  # Simulation

    vol_gt = sim_data["vol_gt"]  # Volume 65L
    rots_gt = sim_data["rots_gt"]  # np.Array

    # Define a precision for this experiment
    dtype = old_sim.dtype

    # Generate data at experiment resolution
    offsets = np.zeros((num_imgs, 2)).astype(dtype)
    amplitudes = np.ones(num_imgs)
    sim = Simulation(L=vol_gt.shape[1], n=old_sim.n, vols=vol_gt,
                     offsets=offsets, unique_filters=old_sim.unique_filters, amplitudes=amplitudes, dtype=dtype)
    sim.noise_adder = SnrNoiseAdder(seed=sim.seed, snr=snr)
    sim.rots = rots_gt  # Use true old rotations
    imgs = sim.images(0, np.inf)

    # Estimate simgma
    sq_sigma = 1 / (1 + snr) * np.mean(np.var(sim.images(0, np.inf).asnumpy(), axis=(1, 2)))
    print("sigma^2 = {}".format(sq_sigma))

    # integrator = TrueRotsIntegrator(rots=rots_gt)
    # integrator = UniformIntegrator(ell_max=4, n=num_imgs)
    # integrator = IcosahedronIntegrator(ell_max=3)
    # integrator = SphDes1821Integrator(ell_max=3)
    integrator = HexacosichoronIntegrator(ell_max=4)

    prob = InsideNormLiftingProblem(imgs=imgs,
                          vol=vol_gt,  # TODO get GT out of here for actual experiments
                          filter=sim.unique_filters[0],
                          integrator=integrator,
                          rots_prior=None  # rots_init,
                          )

    vol_reg = 0.
    dens_reg1 = 1e-8  # was 0.001
    dens_reg2 = 1.

    basis = FBBasis3D((prob.L, prob.L, prob.L))

    def cost_function(problem):
        fidelity_penalty = l2_data_fidelity(problem) / sq_sigma

        vol_l2_penalty = vol_reg * l2_vol_prior(problem)
        dens_l2_penalty = dens_reg1 * l2_dens_prior(problem)
        # if problem.rots_prior_integrands is not None:
        #     dens_prior_penalty = dens_reg2 * integral_dens_prior(problem)
        # else:
        #     dens_prior_penalty = 0.

        cost = fidelity_penalty + vol_l2_penalty + dens_l2_penalty
        logger.info(
            "data penalty = {} | vol_reg penalty = {} | dens_reg1 penalty = {}".format(
                fidelity_penalty,
                vol_l2_penalty,
                dens_l2_penalty)
            )
        return cost

    fvol_init = None  # basis.evaluate_t(vol_gt.asnumpy()[0].T)

    def vol_update(problem, x0=fvol_init):
        return conjugate_gradient_update(problem, basis, x0=x0, regularizer=vol_reg, tol=1e-3, maxiter=10)

    def dens_update(problem):
        quadratic_optimisation_update(problem, sq_sigma=sq_sigma, reg1=dens_reg1, reg2=dens_reg2)
        # problem.rots_dcoef = np.eye(problem.ell, problem.n, dtype=problem.dtype)

    solver = LiftingSolver(problem=prob,
                           cost=cost_function,
                           max_it=1,  # max_it
                           tol=1e-3,
                           vol_update=vol_update,
                           dens_update=dens_update,
                           )

    solver.solve(return_result=False)

    # Save result
    exp.save("solver_data_{}SNR_{}N".format(int(1 / snr), num_imgs),
             # Data
             ("sim", sim),
             ("vol_init", vol_init),  # (img_size,)*3
             # ("rots_init", rots_init),
             ("vol_gt", vol_gt),  # (img_size,)*3
             ("rots_gt", rots_gt),
             # Results
             ("volume_est", solver.problem.vol),
             # ("rots_est", solver.problem.rots),
             ("density_est", [integrator.angles, integrator.coeffs2weights(solver.problem.rots_dcoef)]),
             ("cost", solver.cost),
             # ("relerror_u", solver.relerror_u),
             # ("relerror_g", solver.relerror_g),
             # ("relerror_tot", solver.relerror_tot)
             )
