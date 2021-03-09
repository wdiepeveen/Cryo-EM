"""
This script illustrates the estimation of orientation angles using synchronization
matrix and voting method, based on simulated data projected from a 3D CryoEM map.
"""

import logging

import mrcfile
import numpy as np

from scipy.ndimage import zoom
from scipy.ndimage.filters import gaussian_filter

from aspire.abinitio import CLSyncVoting
from aspire.basis import FBBasis3D
from aspire.operators import RadialCTFFilter
from aspire.reconstruction import MeanEstimator
from aspire.source.simulation import Simulation

from aspire.volume import Volume

from noise.noise import SnrNoiseAdder
from tools.exp_tools import Exp

logger = logging.getLogger(__name__)


def preprocessing(exp=None,
                  num_imgs=None,
                  snr=1.,
                  img_size=65,
                  init_vol_smudge=2.,
                  data_path=None
                  ):

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
    defocus_min = 1.5e4  # Minimum defocus value (in angstroms)
    defocus_max = 2.5e4  # Maximum defocus value (in angstroms)
    defocus_ct = 7  # Number of defocus groups.
    Cs = 2.0  # Spherical aberration
    alpha = 0.1  # Amplitude contrast

    logger.info("Initialize simulation object and CTF filters.")
    # TODO just pick one. We aren't guessing CTF anyways
    # Create CTF filters
    filters = [
        RadialCTFFilter(pixel_size, voltage, defocus=d, Cs=Cs, alpha=alpha)
        for d in np.linspace(defocus_min, defocus_max, defocus_ct)
    ]

    # Load the map file of a 70S Ribosome and downsample the 3D map to desired resolution.
    # The downsampling should be done by the internal function of Volume object in future.
    logger.info(
        f"Load 3D map and downsample 3D map to desired grids "
        f"of {img_size} x {img_size} x {img_size}."
    )
    infile = mrcfile.open(data_path)
    vol_gt = Volume(infile.data.astype(dtype))

    # # Downsample data to correct size
    # vols = vol_gt.downsample((img_size,) * 3)

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

    sim = Simulation(L=img_size, n=num_imgs, vols=exp_vol_gt, unique_filters=filters, dtype=dtype)
    sim.noise_adder = SnrNoiseAdder(seed=sim.seed, snr=snr)


    logger.info("Get true rotation angles generated randomly by the simulation object.")
    rots_gt = sim.rots

    # Initialize an orientation estimation object and perform view angle estimation
    logger.info("Estimate rotation angles using synchronization matrix and voting method.")
    orient_est = CLSyncVoting(sim, n_theta=72)
    orient_est.estimate_rotations()
    rots_est = orient_est.rotations


    # Start reconstruction
    rec_img_size = 33
    vol_gt_ds = vol_gt.downsample((rec_img_size,) * 3)
    prob = Simulation(L=rec_img_size, n=num_imgs, vols=vol_gt_ds, unique_filters=filters, dtype=dtype)
    prob.noise_adder = SnrNoiseAdder(seed=sim.seed, snr=snr)
    prob.rots = rots_est

    # Specify the normal FB basis method for expending the 2D images
    basis = FBBasis3D((rec_img_size, rec_img_size, rec_img_size))

    # Estimate the mean. This uses conjugate gradient on the normal equations for
    # the least-squares estimator of the mean volume. The mean volume is represented internally
    # using the basis object, but the output is in the form of an
    # L-by-L-by-L array.
    mean_estimator = MeanEstimator(prob, basis)

    if init_vol_smudge == 0.:
        mean_est = mean_estimator.estimate()
        vol_init = mean_est
    else:
        mean_est = mean_estimator.estimate(tol=1e-2)
        vol_init = Volume(gaussian_filter(mean_est.asnumpy(), init_vol_smudge))

    vol_init = Volume(zoom(vol_init.asnumpy()[0], img_size/rec_img_size, order=1))

    # Save real vol and save vol from real rots
    exp.save("simulation_data_{}snr_{}n".format(int(1/snr), num_imgs),
             ("sim", sim),
             ("vol_init", vol_init),  # (img_size,)*3
             ("rots_init", rots_est),
             ("vol_gt", exp_vol_gt),  # (img_size,)*3
             ("rots_gt", rots_gt)
             )
