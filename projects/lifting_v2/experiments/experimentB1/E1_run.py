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

from projects.lifting_v2.src.solvers.refinement_solver1 import Refinement_Solver1

logger = logging.getLogger(__name__)


def run_experiment(exp=None,
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

    # Define a precision for this experiment
    dtype = np.float32

    solver_data = exp.open_pkl(data_dir, data_filename)
    # Load data
    old_solver = solver_data["solver"]
    vol_gt = solver_data["vol_gt"]  # Volume 65L
    rots_gt = solver_data["rots_gt"]
    snr = solver_data["SNR"]
    # Load lifting results
    volume_est = old_solver.plan.vol
    rots = old_solver.plan.integrator.rots
    quats = old_solver.plan.integrator.quaternions
    density_est = old_solver.plan.rots_coeffs

    # Up- or downsample data for experiment
    img_size = old_solver.plan.L
    if img_size >= vol_gt.shape[1]:
        if img_size == vol_gt.shape[1]:
            exp_vol_gt = vol_gt
        else:
            exp_vol_gt = Volume(zoom(vol_gt.asnumpy()[0], img_size / vol_gt.shape[1]))  # cubic spline interpolation
    else:
        exp_vol_gt = vol_gt.downsample((img_size,) * 3)

    # TODO not necessary
    rots_indices = np.argmax(density_est, axis=0)
    rots_est = rots[rots_indices]
    # quats_est = quats[rots_indices]

    solver = Refinement_Solver1(quaternions=old_solver.plan.integrator.quaternions,
                                rots_coeffs=old_solver.plan.rots_coeffs,
                                squared_noise_level=old_solver.plan.squared_noise_level,
                                images=old_solver.plan.images,
                                filter=old_solver.plan.filter,
                                amplitude=old_solver.plan.amplitude,
                                volume_reg_param=volume_reg_param,
                                dtype=dtype,
                                seed=0,
                                )

    solver.solve()

    # TODO routine where we also use the obtained rots_est without finding the mean
    # solver2 = Refinement_Solver1(vol=volume_est,
    #                              rots=rots_gt[0:num_imgs],
    #                              squared_noise_level=old_solver.plan.o.squared_noise_level,
    #                              stop=1,
    #                              stop_rots_gd=0,
    #                              gd_step_size=gd_step_size,
    #                              images=images,
    #                              filter=old_solver.plan.p.filter,
    #                              amplitude=old_solver.plan.p.amplitude,
    #                              kernel=kernel,
    #                              integrator=integrator,
    #                              volume_reg_param=volume_reg_param,
    #                              dtype=np.float32,
    #                              seed=0,
    #                              )
    # logger.info("Reconstruction from gt rotations")
    # solver2.solve()

    # Save result
    exp.save("solver_data_{}SNR_{}N".format(int(1 / snr), old_solver.plan.N),
             # Data
             ("solver", solver),
             # ("solver2", solver2),
             ("vol_gt", exp_vol_gt),  # (img_size,)*3
             ("vol_init", volume_est),  # (img_size,)*3
             ("rots_gt", rots_gt),
             ("rots_init", rots_est),
             ("SNR", snr),
             )
