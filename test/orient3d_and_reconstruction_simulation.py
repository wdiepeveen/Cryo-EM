"""
This script illustrates the estimation of orientation angles using synchronization
matrix and voting method, based on simulated data projected from a 3D CryoEM map.
"""

import logging
import os

import mrcfile
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage.filters import gaussian_filter

from aspire.abinitio import CLSyncVoting
from aspire.basis import FBBasis3D
from aspire.image.xform import NoiseAdder
from aspire.operators import RadialCTFFilter,ScalarFilter
from aspire.reconstruction import MeanEstimator
from aspire.source.simulation import Simulation
from aspire.utils.coor_trans import (
    get_aligned_rotations,
    get_rots_mse,
    register_rotations,
)
from aspire.volume import Volume

from noise.noise import SnrNoiseAdder
from tools.exp_tools import Exp

logger = logging.getLogger(__name__)

DATA_DIR = os.path.join(os.path.dirname(__file__), "../data/")
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "../results/")


logger.info(
    "This script illustrates orientation estimation using "
    "synchronization matrix and voting method"
)


# Define a precision for this experiment
dtype = np.float32

# Set the sizes of images
img_size = 33

# Set the total number of images generated from the 3D map
num_imgs= 128

exp = Exp()
exp.begin(prefix="data_simulation", postfix="{}p_{}n".format(img_size,num_imgs))

# Specify the CTF parameters not used for this example
# but necessary for initializing the simulation object
pixel_size = 5  # Pixel size of the images (in angstroms)
voltage = 200  # Voltage (in KV)
defocus_min = 1.5e4  # Minimum defocus value (in angstroms) TODO what is this?
defocus_max = 2.5e4  # Maximum defocus value (in angstroms) TODO and what is this?
defocus_ct = 7  # Number of defocus groups. TODO and this?
Cs = 2.0  # Spherical aberration
alpha = 0.1  # Amplitude contrast

logger.info("Initialize simulation object and CTF filters.")
# Create CTF filters
filters = [
    RadialCTFFilter(pixel_size, voltage, defocus=d, Cs=2.0, alpha=0.1)
    for d in np.linspace(defocus_min, defocus_max, defocus_ct)
]

# Load the map file of a 70S Ribosome and downsample the 3D map to desired resolution.
# The downsampling should be done by the internal function of Volume object in future.
logger.info(
    f"Load 3D map and downsample 3D map to desired grids "
    f"of {img_size} x {img_size} x {img_size}."
)
infile = mrcfile.open(os.path.join(DATA_DIR, "clean70SRibosome_vol_65p.mrc"))
vol_gt = Volume(infile.data.astype(dtype))
vols = vol_gt.downsample((img_size,) * 3)

# Create a simulation object with specified filters and the downsampled 3D map
logger.info("Use downsampled map to creat simulation object.")
snr = 1./64
sim = Simulation(L=img_size, n=num_imgs, vols=vols, unique_filters=filters, dtype=dtype)
sim.noise_adder = SnrNoiseAdder(seed=sim.seed, snr=snr)
print(sim.noise_adder is None)

# Show typical images from simulation
num_plts = 3
images = sim.images(0,num_plts,enable_noise=True)
f, axarr = plt.subplots(1,num_plts)
for i in range(num_plts):
    axarr[i].imshow(images.asnumpy()[i])

plt.show()

logger.info("Get true rotation angles generated randomly by the simulation object.")
rots_gt = sim.rots

# Initialize an orientation estimation object and perform view angle estimation
logger.info("Estimate rotation angles using synchronization matrix and voting method.")
orient_est = CLSyncVoting(sim, n_theta=36)
orient_est.estimate_rotations()
rots_est = orient_est.rotations

# Get register rotations after performing global alignment
Q_mat, flag = register_rotations(rots_est, rots_gt)
regrot = get_aligned_rotations(rots_est, Q_mat, flag)
mse_reg = get_rots_mse(regrot, rots_gt)
logger.info(
    f"MSE deviation of the estimated rotations using register_rotations : {mse_reg}"
)

# Start reconstruction

prob = sim
prob.rots = rots_est

# Specify the normal FB basis method for expending the 2D images
basis = FBBasis3D((img_size, img_size, img_size))

# Estimate the mean. This uses conjugate gradient on the normal equations for
# the least-squares estimator of the mean volume. The mean volume is represented internally
# using the basis object, but the output is in the form of an
# L-by-L-by-L array.

mean_estimator_up = MeanEstimator(prob, basis)
mean_est_up = mean_estimator_up.estimate(tol=1e-2)

filtered_mean_est = gaussian_filter(mean_est_up.asnumpy(), 5)

# Save to output file
exp.save_mrc("reconstructed70SRibosome_vol_{}p".format(img_size), mean_est_up.asnumpy())
exp.save_mrc("filtered_reconstructed70SRibosome_vol_{}p".format(img_size), filtered_mean_est)

# Save real vol and save vol from real rots
exp.save("simulation_data",
         ("sim", sim),
         ("vol_init", mean_est_up),
         ("rot_init", rots_est),
         ("vol_gt", vol_gt),
         ("rots_gt", rots_gt)
         )
