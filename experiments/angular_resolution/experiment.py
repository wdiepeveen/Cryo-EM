import logging
import numpy as np

import mrcfile

from aspire.operators import RadialCTFFilter
from aspire.source.simulation import Simulation
from aspire.volume import Volume

# Own functions
from solvers.lifting.functions.cost_functions import l2_data_fidelity_on, l2_vol_prior, l2_dens_prior
from noise.noise import SnrNoiseAdder
from solvers.lifting.integration.hexacosichoron import HexacosichoronIntegrator
from solvers.lifting.integration.sd1821 import SphDes1821Integrator
from solvers.lifting.integration.almost_true_rots import AlmostTrueRotsIntegrator
from solvers.lifting.update.density.qo_on import quadratic_optimisation_update
from solvers.lifting.update.volume.exact_on import exact_update
from solvers.lifting.problems.outside_norm import OutsideNormLiftingProblem
from solvers.lifting.solver import LiftingSolver

logger = logging.getLogger(__name__)


def experiment(exp=None,
               ell_max=10,
               num_imgs=None,
               snr=1.,
               img_size=65,
               data_path=None):
    logger.info(
        "This experiment angular resolution needed for refinement using "
        "a orientation lifting approach"
    )

    # Define a precision for this experiment
    dtype = np.float32

    # Specify the CTF parameters not used for this example
    # but necessary for initializing the simulation object
    pixel_size = 5  # Pixel size of the images (in angstroms)
    voltage = 200  # Voltage (in KV)
    defocus = 1.5e4  # Minimum defocus value (in angstroms)
    Cs = 2.0  # Spherical aberration
    alpha = 0.1  # Amplitude contrast

    logger.info("Initialize simulation object and CTF filters.")
    # Create CTF filters
    filters = [
        RadialCTFFilter(pixel_size, voltage, defocus=defocus, Cs=Cs, alpha=alpha)
    ]

    # Load the map file of a 70S Ribosome and downsample the 3D map to desired resolution.
    # The downsampling should be done by the internal function of Volume object in future.
    logger.info(
        f"Load 3D map and downsample 3D map to desired grids "
        f"of {img_size} x {img_size} x {img_size}."
    )
    infile = mrcfile.open(data_path)
    vol_gt = Volume(infile.data.astype(dtype))

    # Create simulation
    offsets = np.zeros((num_imgs, 2)).astype(dtype)
    amplitudes = np.ones(num_imgs)
    sim = Simulation(L=img_size, n=num_imgs, vols=vol_gt,
                     offsets=offsets, unique_filters=filters, amplitudes=amplitudes, dtype=dtype)
    sim.noise_adder = SnrNoiseAdder(seed=sim.seed, snr=snr)

    logger.info("Get true rotation angles generated randomly by the simulation object.")
    rots_gt = sim.rots
    imgs = sim.images(0, np.inf)

    # Estimate sigma
    sq_sigma = 1 / (1 + snr) * np.mean(np.var(sim.images(0, np.inf).asnumpy(), axis=(1, 2)))
    print("sigma^2 = {}".format(sq_sigma))

    integrator = AlmostTrueRotsIntegrator(ell_max=ell_max, rots=rots_gt)

    prob = OutsideNormLiftingProblem(imgs=imgs,
                                     vol=vol_gt,  # We are not using volume here at all so no need to change this
                                     filter=sim.unique_filters[0],
                                     integrator=integrator
                                     )

    vol_reg = 1e10
    dens_reg = 1e-8  # was 0.001

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
        return None

    def dens_update(problem):
        problem.rots_dcoef = problem.integrator.coeffs

    solver = LiftingSolver(problem=prob,
                           cost=cost_function,
                           max_it=1,  # max_it
                           tol=1e-3,
                           vol_update=vol_update,
                           dens_update=dens_update,
                           )

    solver.solve(return_result=False)



    # Save result
    exp.save("solver_data_{}ell_{}SNR_{}N".format(ell_max, int(1 / snr), num_imgs),
             # Data
             ("images", imgs),
             ("vol_gt", vol_gt),  # (img_size,)*3
             ("angles_gt", integrator.angles),
             # Results
             ("volume_est", solver.problem.vol),
             ("density_est", integrator.coeffs2weights(solver.problem.rots_dcoef))
             )
