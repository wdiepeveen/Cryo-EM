import logging
import numpy as np
import mrcfile

from scipy.ndimage import zoom

from aspire.operators import RadialCTFFilter
from aspire.source.simulation import Simulation
from aspire.volume import Volume

from noise.noise import SnrNoiseAdder
from tools.exp_tools import Exp

from projects.lifting_v2.src.integrators.base.sd1821mrx import SD1821MRx
from projects.lifting_v2.src.solvers.lifting_solver2 import Lifting_Solver2
from projects.lifting_v2.src.solvers.refinement_solver2 import Refinement_Solver2
from projects.lifting_v2.src.util.rots_container import RotsContainer

logger = logging.getLogger(__name__)


def run_experiment(exp=None,
                   max_iter=1,
                   num_imgs=None,
                   snr=None,
                   img_size=None,
                   mr_repeat=None,
                   rots_reg_param=None,
                   rots_reg_scaling_param=66 / 100,  # eta
                   data_path=None,
                   ):
    logger.info(
        "This experiment illustrates orientation refinement using a lifting approach"
    )

    if not isinstance(exp, Exp):
        raise RuntimeError("Cannot run experiment without Exp object")

    if data_path is None:
        raise RuntimeError("No data path provided")

    # Define a precision for this experiment
    # dtype = np.float32
    dtype = np.float64

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

    # Up- or downsample data for experiment
    if img_size >= vol_gt.shape[1]:
        if img_size == vol_gt.shape[1]:
            exp_vol_gt = vol_gt
        else:
            exp_vol_gt = Volume(zoom(vol_gt.asnumpy()[0], img_size / vol_gt.shape[1]))  # cubic spline interpolation
    else:
        exp_vol_gt = vol_gt.downsample((img_size,) * 3)

    # Create a simulation object with specified filters and the downsampled 3D map
    logger.info("Use downsampled map to creat simulation object.")

    offsets = np.zeros((num_imgs, 2)).astype(dtype)
    amplitudes = np.ones(num_imgs)
    sim = Simulation(L=img_size, n=num_imgs, vols=exp_vol_gt,
                     offsets=offsets, unique_filters=filters, amplitudes=amplitudes, dtype=dtype)
    sim.noise_adder = SnrNoiseAdder(seed=sim.seed, snr=snr)

    logger.info("Get true rotation angles generated randomly by the simulation object.")
    rots_gt = RotsContainer(num_imgs, dtype=dtype)
    rots_gt.rots = sim.rots

    # Estimate sigma
    squared_noise_level = 1 / (1 + snr) * np.sum(np.var(sim.images(0, np.inf).asnumpy(), axis=(1, 2)))
    print("sigma = {}".format(squared_noise_level))
    sigmas = squared_noise_level * np.ones((num_imgs,))
    # tau = 1e-9  # np.sum(vol_init.asnumpy() ** 2)  # / (img_size ** 3)
    tau = np.sum(exp_vol_gt.asnumpy() ** 2)
    print("tau = {}".format(tau))

    integrator = SD1821MRx(repeat=mr_repeat, dtype=dtype)

    solver = Lifting_Solver2(vol=exp_vol_gt,
                             squared_noise_level=squared_noise_level,
                             volume_reg_param=tau,
                             images=sim.images(0, np.inf),
                             filter=sim.unique_filters[0],
                             integrator=integrator,
                             rots_reg_param=rots_reg_param,
                             rots_reg_scaling_param=rots_reg_scaling_param,
                             max_iter=max_iter,
                             save_iterates=True,
                             dtype=dtype,
                             experiment="consistency"
                             )

    solver.solve()

    # Stage 2: Refinement
    rots_indices = np.argmax(solver.plan.rots_coeffs, axis=0)
    rots_init = RotsContainer(num_imgs, dtype=dtype)
    rots_init.rots = solver.plan.integrator.rots[rots_indices]

    refinement_solver = Refinement_Solver2(quaternions=solver.plan.integrator.quaternions,
                                           rots_coeffs=solver.plan.rots_coeffs,
                                           sigmas=solver.plan.sigmas,
                                           tau=solver.plan.tau,
                                           images=solver.plan.images,
                                           filter=solver.plan.filter,
                                           amplitude=solver.plan.amplitude,
                                           dtype=dtype,
                                           seed=0,
                                           )

    refinement_solver.solve()

    # Save result
    exp.save("solver_data_r{}".format(mr_repeat),
             # Data
             ("SNR", snr),
             ("solver", solver),
             ("refinement_solver", refinement_solver),
             ("vol_gt", exp_vol_gt),  # (img_size,)*3
             ("rots_gt", rots_gt),
             ("rots_init", rots_init),
             )
