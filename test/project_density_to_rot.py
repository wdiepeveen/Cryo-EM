from solvers.lifting.integration.almost_true_rots import AlmostTrueRotsIntegrator

# Setup rotations

import logging

import mrcfile
import numpy as np
import os

from scipy.ndimage import zoom
from scipy.ndimage.filters import gaussian_filter
from scipy.spatial.transform import Rotation as R

from aspire.abinitio import CLSyncVoting
from aspire.basis import FBBasis3D
from aspire.operators import RadialCTFFilter
from aspire.reconstruction import MeanEstimator
from aspire.source.simulation import Simulation

from aspire.volume import Volume

import matplotlib.pyplot as plt

from solvers.lifting.update.rots.project import proj

from noise.noise import SnrNoiseAdder
from tools.exp_tools import Exp

logger = logging.getLogger(__name__)

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
    RadialCTFFilter(pixel_size, voltage, defocus=defocus, Cs=2.0, alpha=0.1)
]

img_size = 33
num_imgs = 512

# Set data path
data_dir = "data"
data_filename = "clean70SRibosome_vol_65p.mrc"
data_path = os.path.join("..", data_dir, data_filename)

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

offsets = np.zeros((num_imgs, 2)).astype(dtype)
amplitudes = np.ones(num_imgs)
sim = Simulation(L=img_size, n=num_imgs, vols=exp_vol_gt,
                 offsets=offsets, unique_filters=filters, amplitudes=amplitudes, dtype=dtype)
# sim.noise_adder = SnrNoiseAdder(seed=sim.seed, snr=snr)


logger.info("Get true rotation angles generated randomly by the simulation object.")
rots_gt = sim.rots

# Construct fake density
ell_max = 15
integrator = AlmostTrueRotsIntegrator(ell_max=ell_max, rots=rots_gt)

density = integrator.coeffs2weights(integrator.coeffs)

# Plot refined points
angles = integrator.angles

x = angles[:, 0]
y = angles[:, 1]
z = angles[:, 2]
c = density[0, :] * len(x)  # only first density for visualization
print("Number of angles = {}".format(len(x)))

# Get density plot
fig = plt.figure()
ax = fig.gca(projection='3d')

img = ax.scatter(x, y, z, c=c, cmap=plt.cool())
ax.set_xlabel("$\phi$")
ax.set_ylabel("$\\theta$")
ax.set_zlabel("$\psi$")  #, rotation=0)
plt.colorbar(img)

plt.show()

# Project density
quats = integrator.proj(integrator.coeffs)

points = R.from_quat(quats)
angles = points.as_euler("ZYZ").astype(integrator.dtype)

# Plot rot

xx = np.hstack([x, angles[0,0]])
yy = np.hstack([y, angles[0,1]])
zz = np.hstack([z, angles[0,2]])
cc = np.hstack([c,200])

# Get density plot
fig = plt.figure()
ax = fig.gca(projection='3d')

img = ax.scatter(xx, yy, zz, c=cc, cmap=plt.cool())
ax.set_xlabel("$\phi$")
ax.set_ylabel("$\\theta$")
ax.set_zlabel("$\psi$")  #, rotation=0)
plt.colorbar(img)

plt.show()