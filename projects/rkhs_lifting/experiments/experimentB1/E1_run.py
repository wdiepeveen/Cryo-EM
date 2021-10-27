import logging
import numpy as np
import mrcfile

from scipy.ndimage import zoom

from aspire.operators import RadialCTFFilter
from aspire.source.simulation import Simulation
from aspire.volume import Volume

from noise.noise import SnrNoiseAdder
from tools.exp_tools import Exp

from projects.rkhs_lifting.src.integrators.base.sd1821 import SD1821
from projects.rkhs_lifting.src.integrators.base.sd1821mrx import SD1821MRx
# from projects.rkhs_lifting.src.integrators.base.refined_sd import Refined_SD
from projects.rkhs_lifting.src.integrators import RKHS_Density_Integrator
from projects.rkhs_lifting.src.kernels.rescaled_cosine import Rescaled_Cosine_Kernel
from projects.rkhs_lifting.src.solvers.lifting_solver2 import RKHS_Lifting_Solver2
from projects.rkhs_lifting.src.solvers.refinement_solver1 import Refinement_Solver1

logger = logging.getLogger(__name__)


def run_experiment(exp=None,
                   num_imgs=None,
                   snr=1.,
                   img_size=65,
                   kernel_radius=np.pi / 20,
                   l=3,
                   data_dir=None,
                   data_filename=None,
                   ):
    logger.info(
        "This experiment illustrates orientation refinement using a lifting approach"
    )

    if not isinstance(exp, Exp):
        raise RuntimeError("Cannot run experiment without Exp object")

    if data_dir is None or data_filename is None:
        raise RuntimeError("No data path provided")

    # Define a precision for this experiment
    dtype = np.float32

    solver_data = exp.open_pkl(data_dir, data_filename)
    # Load data
    # sim = solver_data["sim"]  # Simulator
    vol_gt = solver_data["vol_gt"]  # Volume 65L
    rots_gt = solver_data["rots_gt"]
    squared_noise_level = solver_data["squared_noise_level"]
    filter_ = solver_data["filter"]
    images = solver_data["images"]
    # Load lifting results
    volume_est = solver_data["volume_est"]
    rots = solver_data["rots"]
    density_est = solver_data["density_on_angles"]

    rots_indices = np.argmax(density_est, axis=0)
    rots_est = rots[rots_indices]

    # Load the map file of a 70S Ribosome and downsample the 3D map to desired resolution.
    # The downsampling should be done by the internal function of Volume object in future.
    logger.info(
        f"Load 3D map and downsample 3D map to desired grids "
        f"of {img_size} x {img_size} x {img_size}."
    )

    # Up- or downsample data for experiment
    if img_size >= volume_est.shape[1]:
        if img_size == volume_est.shape[1]:
            exp_vol_gt = volume_est
        else:
            exp_vol_gt = Volume(zoom(volume_est.asnumpy()[0], img_size / volume_est.shape[1]))  # cubic spline interpolation
    else:
        exp_vol_gt = volume_est.downsample((img_size,) * 3)

    # Estimate simgma
    squared_noise_level = 1 / (1 + snr) * np.mean(np.var(sim.images(0, np.inf).asnumpy(), axis=(1, 2)))
    print("sigma^2 = {}".format(squared_noise_level))

    refined_integrator = SD1821MRx(repeat=mr_repeat, dtype=dtype)
    # resolution = refined_integrator.mesh_norm
    radius = kernel_radius
    kernel = Rescaled_Cosine_Kernel(quaternions=refined_integrator.quaternions, radius=radius, dtype=dtype)

    rkhs_integrator = RKHS_Density_Integrator(base_integrator=refined_integrator, kernel=kernel, dtype=dtype)
    logger.info(
        "integrator separation distance = {}, corresponding to k = {}".format(rkhs_integrator.base_integrator.sep_dist,
                                                                              np.pi / rkhs_integrator.base_integrator.sep_dist))

    solver = Refinement_Solver1(vol=volume_est,
                                rots=rots_est,
                                squared_noise_level=squared_noise_level,
                                stop=1,
                                stop_rots_gd=5,
                                images=images,  # TODO from data loader
                                filter=filter_,
                                amplitude=None,  # TODO hard code
                                kernel=None,  # TODO as input function
                                integrator=None,
                                dtype=np.float32,
                                seed=0,
                                )

    solver.solve()

    # exp.save_npy("rotation_density_coeffs", solver.problem.rots_dcoef)

    # Postprocessing step
    # TODO fix
    # quats = solver.problem.integrator.proj(solver.problem.rots_dcoef)
    #
    # refined_rots = quat2mat(quats)

    # refined_vol = solver.problem.vol
    # refined_vol = exact_refinement(solver.problem, refined_rots)
    # TODO maybe just do a refinement solver instead of this (although might be a unnecessary work)

    # Save result
    exp.save("solver_data_{}SNR_{}N".format(int(1 / snr), num_imgs),
             # Data
             ("sim", sim),  # TODO: don't save sim here, but the clean and noisy images
             ("vol_gt", exp_vol_gt),  # (img_size,)*3
             ("rots_gt", rots_gt),
             # Results
             ("volume_est", solver.plan.o.vol),
             # ("refined_volume_est", refined_vol),
             # ("rots_est", solver.problem.rots),
             ("angles", solver.plan.p.integrator.angles),
             ("density_on_angles", solver.plan.p.integrator.coeffs_to_weights(solver.plan.o.density_coeffs)),
             # ("refined_rots_est", refined_rots),
             ("cost", solver.cost),
             # ("relerror_u", solver.relerror_u),
             # ("relerror_g", solver.relerror_g),
             # ("relerror_tot", solver.relerror_tot)
             )
