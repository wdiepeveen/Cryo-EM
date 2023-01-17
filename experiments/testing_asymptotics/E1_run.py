import logging
import numpy as np
import mrcfile

from scipy.ndimage.filters import gaussian_filter

from aspire.operators import RadialCTFFilter
from aspire.source.simulation import Simulation
from aspire.volume import Volume

from noise.noise import SnrNoiseAdder
from tools.exp_tools import Exp

from src.integrators.sd1821mrx import SD1821MRx
from src.solvers.lifting_solver import Lifting_Solver
from src.util.rots_container import RotsContainer

logger = logging.getLogger(__name__)


def run_experiment(exp=None,
                   num_imgs=1024,
                   snr=1.,
                   mr_repeat=1,
                   rots_reg_param=None,
                   eta_range=None,
                   # rots_reg_scaling_param=66 / 100,  # eta
                   data_path=None,
                   vol_smudge=0,
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

    # Load the map file
    infile = mrcfile.open(data_path)
    vol_gt = Volume(infile.data.astype(dtype))
    img_size = vol_gt.shape[1]

    logger.info(
        f"Load 3D map and downsample 3D map to desired grids "
        f"of {img_size} x {img_size} x {img_size}."
    )

    # Specify the CTF parameters not used for this example
    # but necessary for initializing the simulation object
    pixel_size = infile.voxel_size.tolist()[0]  # Pixel size of the images (in angstroms)
    voltage = 200  # Voltage (in KV)
    defocus = 1.5e4  # Minimum defocus value (in angstroms)
    Cs = 2.0  # Spherical aberration
    alpha = 0.1  # Amplitude contrast

    logger.info("Initialize simulation object and CTF filters.")
    # Create CTF filters
    filters = [
        RadialCTFFilter(pixel_size, voltage, defocus=defocus, Cs=Cs, alpha=alpha)
    ]

    vol_init = vol_gt
    if vol_smudge > 0:
        logger.info("Smudging volume")
        vol_init = Volume(gaussian_filter(vol_gt.asnumpy()[0], vol_smudge))

    # Create a simulation object with specified filters and the downsampled 3D map
    logger.info("Use downsampled map to creat simulation object.")

    offsets = np.zeros((num_imgs, 2)).astype(dtype)
    amplitudes = np.ones(num_imgs)
    sim = Simulation(L=img_size, n=num_imgs, vols=vol_gt,
                     offsets=offsets, unique_filters=filters, amplitudes=amplitudes, dtype=dtype)
    sim.noise_adder = SnrNoiseAdder(seed=sim.seed, snr=snr)

    logger.info("Get true rotation angles generated randomly by the simulation object.")
    rots_gt = RotsContainer(num_imgs, dtype=dtype)
    rots_gt.rots = sim.rots

    # Estimate sigma
    squared_noise_level = np.mean(np.var(sim.images(0, np.inf).asnumpy(), axis=(1, 2)))
    tau = np.mean(vol_gt.asnumpy() ** 2)
    tau_init = np.mean(vol_gt.asnumpy()[0] ** 2)

    integrator = SD1821MRx(repeat=mr_repeat, dtype=dtype)

    solver = Lifting_Solver(vol=vol_init,
                            squared_noise_level=squared_noise_level,
                            volume_reg_param=tau,
                            images=sim.images(0, np.inf),
                            filter=sim.unique_filters[0],
                            integrator=integrator,
                            # rots_reg_scaling_param=rots_reg_scaling_param,
                            max_iter=1,
                            save_iterates=True,
                            dtype=dtype,
                            )

    solver.plan.lambd = rots_reg_param * np.ones(num_imgs)
    solver.plan.data_discrepancy_update()

    # rather than solve the whole thing, why not just update several parameters and only do steps of the solver
    for eta in eta_range:
        print("eta = {}".format(eta))
        solver.plan.eta = eta
        solver.rots_step()

        # Save result
        exp.save_npy("rots_est_eta{}".format(int(eta * 100)), solver.plan.quaternions)
        exp.save_npy("rots_coeffs_eta{}".format(int(eta * 100)), solver.plan.rots_coeffs)

    # Stage 2: Refinement
    exp.save_npy("rots_gt", rots_gt.quaternions)
    # Note that the rot with highest coefficient is invariant under gamma and eta
    rots_indices = np.argmax(solver.plan.rots_coeffs, axis=0)
    rots_init = RotsContainer(num_imgs, dtype=dtype)
    rots_init.rots = solver.plan.integrator.rots[rots_indices]
    exp.save_npy("rots_init", rots_init.quaternions)

    exp.save("solver_data_r{}".format(mr_repeat),
             # Data
             ("vol_gt", vol_gt),  # (img_size,)*3
             ("voxel_size", infile.voxel_size),
             ("vol_init", vol_init),
             ("SNR", snr),
             ("images", solver.plan.images),
             )
