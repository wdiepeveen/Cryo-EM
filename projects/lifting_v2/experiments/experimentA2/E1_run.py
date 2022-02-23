import logging
import numpy as np
import mrcfile

from scipy.ndimage import zoom

from aspire.operators import RadialCTFFilter
from aspire.source.simulation import Simulation
from aspire.volume import Volume

from noise.noise import SnrNoiseAdder
from tools.exp_tools import Exp

from projects.lifting_v2.src.integrators.base.sd1821 import SD1821
from projects.lifting_v2.src.integrators.base.sd1821mrx import SD1821MRx
# from projects.lifting_v2.src.integrators.base.refined_sd import Refined_SD
from projects.lifting_v2.src.integrators.double import Double_SO3_Integrator
from projects.lifting_v2.src.kernels.function.rescaled_cosine import Rescaled_Cosine_Kernel
from projects.lifting_v2.src.solvers.lifting_solver1 import Lifting_Solver1

logger = logging.getLogger(__name__)


def run_experiment(exp=None,
                   num_imgs=None,
                   snr=1.,
                   img_size=65,
                   kernel_radius=np.pi / 20,
                   mr_repeat=1,
                   volume_reg_param=1e10,  # \lambda1
                   rots_coeffs_reg_param=1e-6,  # \lambda2
                   rots_coeffs_reg_scaling_param=66 / 100,  # p
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
    rots_gt = sim.rots

    # Estimate simgma
    squared_noise_level = 1 / (1 + snr) * np.mean(np.var(sim.images(0, np.inf).asnumpy(), axis=(1, 2)))
    print("sigma^2 = {}".format(squared_noise_level))

    refined_integrator = SD1821MRx(repeat=mr_repeat, dtype=dtype)
    # resolution = refined_integrator.mesh_norm
    radius = kernel_radius
    kernel = Rescaled_Cosine_Kernel(outer_quaternions=refined_integrator.quaternions, radius=radius, dtype=dtype)
    # TODO add kernel to integrator if we want to test it
    integrator = Double_SO3_Integrator(outer_quaternions=refined_integrator.quaternions)

    # rkhs_integrator = RKHS_Density_Integrator(base_integrator=refined_integrator, kernel=kernel, dtype=dtype)
    # logger.info(
    #     "integrator separation distance = {}, corresponding to k = {}".format(rkhs_integrator.base_integrator.sep_dist,
    #                                                                           np.pi / rkhs_integrator.base_integrator.sep_dist))

    solver = Lifting_Solver1(vol=exp_vol_gt,
                             max_iter=1,
                             squared_noise_level=squared_noise_level,
                             images=sim.images(0, np.inf),
                             filter=sim.unique_filters[0],
                             integrator=integrator,
                             volume_reg_param=volume_reg_param,
                             rots_coeffs_reg_param=rots_coeffs_reg_param,
                             rots_coeffs_reg_scaling_param=rots_coeffs_reg_scaling_param,
                             dtype=dtype,
                             )

    solver.solve()

    # Save result
    exp.save("solver_data_{}SNR_{}N".format(int(1 / snr), num_imgs),
             # Data
             ("solver", solver),  # TODO check whether this works at all
             ("sim", sim),  # TODO: don't save sim here, but the clean and noisy images
             ("vol_gt", exp_vol_gt),  # (img_size,)*3
             ("rots_gt", rots_gt),
             # Results
             ("volume_est", solver.plan.vol),
             # ("refined_volume_est", refined_vol),
             # ("rots_est", solver.problem.rots),
             ("angles", solver.plan.integrator.angles),
             ("rots", solver.plan.integrator.rots),
             ("quats", solver.plan.integrator.quaternions),
             # ("density_on_angles", solver.plan.p.integrator.coeffs_to_weights(solver.plan.o.density_coeffs)),
             ("coeffs_on_angles", solver.plan.rots_coeffs),
             # ("refined_rots_est", refined_rots),
             ("cost", solver.cost),
             # ("relerror_u", solver.relerror_u),
             # ("relerror_g", solver.relerror_g),
             # ("relerror_tot", solver.relerror_tot)
             # ("squared_noise_level", squared_noise_level),
             # ("filter", solver.plan.p.filter),
             ("images", solver.plan.images),
             ("SNR",snr)
             )