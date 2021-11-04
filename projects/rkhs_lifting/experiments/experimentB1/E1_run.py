import logging
import numpy as np
import mrcfile

from scipy.ndimage import zoom

from aspire.operators import RadialCTFFilter
from aspire.source.simulation import Simulation
from aspire.volume import Volume
from aspire.image.image import Image

from noise.noise import SnrNoiseAdder
from tools.exp_tools import Exp

from projects.rkhs_lifting.src.integrators.base.local_regular import Local_Regular
from projects.rkhs_lifting.src.integrators import RKHS_Density_Integrator
from projects.rkhs_lifting.src.kernels.rescaled_cosine import Rescaled_Cosine_Kernel
from projects.rkhs_lifting.src.solvers.lifting_solver2 import RKHS_Lifting_Solver2
from projects.rkhs_lifting.src.solvers.refinement_solver1 import Refinement_Solver1

logger = logging.getLogger(__name__)


def run_experiment(exp=None,
                   num_imgs=None,
                   snr=1.,
                   img_size=65,
                   kernel_radius=np.pi / 90,
                   l=3,
                   integrator_radius=np.pi / 180,
                   stop_rots_gd=5,
                   gd_step_size=None,
                   volume_reg_param=None,
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

    assert integrator_radius <= kernel_radius

    # Define a precision for this experiment
    dtype = np.float32

    solver_data = exp.open_pkl(data_dir, data_filename)
    # Load data
    old_solver = solver_data["solver"]
    vol_gt = solver_data["vol_gt"]  # Volume 65L
    rots_gt = solver_data["rots_gt"]
    # Load lifting results
    volume_est = old_solver.plan.o.vol
    rots = old_solver.plan.p.integrator.rots
    quats = old_solver.plan.p.integrator.quaternions
    density_est = old_solver.plan.o.density_coeffs

    rots_indices = np.argmax(density_est, axis=0)
    rots_est = rots[rots_indices]
    # quats_est = quats[rots_indices]

    # Load the map file of a 70S Ribosome and downsample the 3D map to desired resolution.
    # The downsampling should be done by the internal function of Volume object in future.
    logger.info(
        f"Load 3D map and downsample 3D map to desired grids "
        f"of {img_size} x {img_size} x {img_size}."
    )

    # Up- or downsample data for experiment  # TODO see whether this makes sense -> not just GT here?
    if img_size >= volume_est.shape[1]:
        if img_size == volume_est.shape[1]:
            exp_vol_gt = volume_est
        else:
            exp_vol_gt = Volume(
                zoom(volume_est.asnumpy()[0], img_size / volume_est.shape[1]))  # cubic spline interpolation
    else:
        exp_vol_gt = volume_est.downsample((img_size,) * 3)

    images = Image(old_solver.plan.p.images[0:num_imgs])
    integrator = Local_Regular(l=l, radius=integrator_radius, dtype=dtype)
    kernel = Rescaled_Cosine_Kernel(radius=kernel_radius, dtype=dtype)

    solver = Refinement_Solver1(vol=volume_est,
                                rots=rots_est,
                                squared_noise_level=old_solver.plan.o.squared_noise_level,
                                stop=1,
                                stop_rots_gd=stop_rots_gd,
                                gd_step_size=gd_step_size,
                                images=images,
                                filter=old_solver.plan.p.filter,
                                amplitude=old_solver.plan.p.amplitude,
                                kernel=kernel,
                                integrator=integrator,
                                volume_reg_param=volume_reg_param,
                                dtype=np.float32,
                                seed=0,
                                )

    solver.solve()

    solver2 = Refinement_Solver1(vol=volume_est,
                                rots=rots_gt,
                                squared_noise_level=old_solver.plan.o.squared_noise_level,
                                stop=1,
                                stop_rots_gd=0,
                                gd_step_size=gd_step_size,
                                images=images,
                                filter=old_solver.plan.p.filter,
                                amplitude=old_solver.plan.p.amplitude,
                                kernel=kernel,
                                integrator=integrator,
                                volume_reg_param=volume_reg_param,
                                dtype=np.float32,
                                seed=0,
                                )
    logger.info("Reconstruction from gt rotations")
    solver2.solve()

    # Save result
    exp.save("solver_data_{}SNR_{}N".format(int(1 / snr), num_imgs),
             # Data
             ("solver", solver),
             ("solver2", solver2),
             ("vol_gt", exp_vol_gt),  # (img_size,)*3
             ("vol_init", volume_est),  # (img_size,)*3
             ("rots_gt", rots_gt),
             ("rots_init", rots_est),
             )
