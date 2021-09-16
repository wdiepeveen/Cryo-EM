import logging
import numpy as np
import matplotlib.pyplot as plt

from scipy.spatial.transform import Rotation as R

from aspire.source.simulation import Simulation

# Own functions
from solvers.lifting.functions.cost_functions import l2_data_fidelity_on, l2_vol_prior, l2_dens_prior
from noise.noise import SnrNoiseAdder
from solvers.lifting.integration.hexacosichoron import HexacosichoronIntegrator
from solvers.lifting.integration.uniform import UniformIntegrator
from solvers.lifting.integration.sd1821 import SphDes1821Integrator
from solvers.lifting.integration.almost_true_rots import AlmostTrueRotsIntegrator
from solvers.lifting.integration.refined_mesh import RefinedMeshIntegrator
from solvers.lifting.update.density.pdhg_on_2 import primal_dual_hybrid_gradient_update
from solvers.lifting.update.volume.exact_on import exact_update
from solvers.lifting.update.volume.exact import exact_refinement
from solvers.lifting.problems.primal_dual_outside_norm import PrimalDualOutsideNormLiftingProblem
from solvers.lifting.solver import LiftingSolver

from solvers.lifting.functions.rot_converters import quat2mat

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
    # integrator = UniformIntegrator(ell_max=5, n=3000)
    # integrator = IcosahedronIntegrator(ell_max=3)
    integrator = SphDes1821Integrator(ell_max=8)
    # integrator = RefinedMeshIntegrator(ell_max=15, mesh_norm=np.pi/20)
    # integrator = RefinedMeshIntegrator(ell_max=8, mesh_norm=np.pi/10, base_integrator="icosahedron")
    # integrator = HexacosichoronIntegrator(ell_max=5)
    # integrator = AlmostTrueRotsIntegrator(ell_max=18, rots=rots_gt)

    prob = PrimalDualOutsideNormLiftingProblem(imgs=imgs,
                                               vol=vol_gt,  # TODO get GT out of here for actual experiments
                                               filter=sim.unique_filters[0],
                                               integrator=integrator
                                               )

    vol_reg = 1e10
    dens_reg = 1e-5  # was 0.001

    # basis = FBBasis3D((prob.L, prob.L, prob.L))

    def cost_function(problem):
        fidelity_penalty = l2_data_fidelity_on(problem) / sq_sigma

        vol_l2_penalty = vol_reg * l2_vol_prior(problem)
        dens_l2_penalty = dens_reg * l2_dens_prior(problem)

        cost = fidelity_penalty + vol_l2_penalty + dens_l2_penalty
        logger.info(
            "data penalty = {} | vol_reg penalty = {} | dens_reg1 penalty = {}".format(
                fidelity_penalty,
                vol_l2_penalty,
                dens_l2_penalty)
        )
        return cost

    def vol_update(problem):
        exact_update(problem, sq_sigma=sq_sigma, regularizer=vol_reg)

    def dens_update(problem):
        # TODO if we actually already have a proper solution from a previous iteration, we want to use that one!!!
        regularizers = np.logspace(-1,np.log(dens_reg)/np.log(10),10,base=10)
        Relerrors = [1.]
        Error0 = 1.
        for i,regularizer in enumerate(regularizers):
            print("================== REGULARIZER {} ==================".format(i))
            print("Regularizer = {}".format(regularizer))
            # TODO optie dat we ook hier al een max iterations en een max tolerance kunnen zetten

            relerrors = primal_dual_hybrid_gradient_update(problem, sq_sigma=sq_sigma, regularizer=regularizer)
            Relerrors += relerrors

        num_its = len(Relerrors)

        plt.figure()
        plt.plot(np.arange(num_its) + 1, Relerrors)
        plt.yscale('linear')
        exp.save_fig("pdhg_relerrors")

        # relerrors = primal_dual_hybrid_gradient_update(problem, sq_sigma=sq_sigma, regularizer=dens_reg)
        #
        # num_its = len(relerrors)
        #
        # plt.figure()
        # plt.plot(np.arange(num_its) + 1, relerrors)
        # plt.yscale('linear')
        # exp.save_fig("pdhg_relerrors")

        # problem.rots_dcoef = np.eye(problem.ell, problem.n, dtype=problem.dtype)
        # problem.rots_dcoef = problem.integrator.coeffs

    solver = LiftingSolver(problem=prob,
                           cost=cost_function,
                           max_it=1,  # max_it
                           tol=1e-3,
                           vol_update=vol_update,
                           dens_update=dens_update,
                           )

    solver.solve(return_result=False)

    exp.save_npy("rotation_density_coeffs", solver.problem.rots_dcoef)

    # Postprocessing step
    # TODO fix
    quats = solver.problem.integrator.proj(solver.problem.rots_dcoef)
    #
    refined_rots = quat2mat(quats)

    refined_vol = solver.problem.vol
    # refined_vol = exact_refinement(solver.problem, refined_rots)
    # TODO maybe just do a refinement solver instead of this (although might be a unnecessary work)

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
             ("refined_volume_est", refined_vol),
             # ("rots_est", solver.problem.rots),
             ("density_est", [integrator.angles, integrator.coeffs2weights(solver.problem.rots_dcoef)]),
             ("refined_rots_est", refined_rots),
             ("cost", solver.cost),
             # ("relerror_u", solver.relerror_u),
             # ("relerror_g", solver.relerror_g),
             # ("relerror_tot", solver.relerror_tot)
             )
