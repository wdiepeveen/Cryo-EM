import logging
import os

import mrcfile
import numpy as np

from aspire.operators import RadialCTFFilter
from aspire.source.simulation import Simulation
from aspire.volume import Volume

from noise.noise import SnrNoiseAdder

from solvers.lifting.integration.icosahedron import IcosahedronIntegrator
from solvers.lifting.problems.inside_norm import LiftingProblem
# from solvers.lifting.update.volume.cg import LeastSquaresCGUpdate
from solvers.lifting.update.density.qo import quadratic_optimisation_update

logger = logging.getLogger(__name__)

DATA_DIR = os.path.join(os.path.dirname(__file__), "../data/")

logger.info(
    "This script illustrates orientation estimation using "
    "synchronization matrix and voting method"
)


# Define a precision for this experiment
dtype = np.float32

# Set the sizes of images
img_size = 33

# Set the total number of images generated from the 3D map
num_imgs= 60  # 12

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
snr = 1./2
# TODO make sure that there are no shifts, amplitudes
offsets = np.zeros((num_imgs, 2)).astype(dtype)
amplitudes = np.ones(num_imgs)
sim = Simulation(L=img_size, n=num_imgs, vols=vols, unique_filters=filters, dtype=dtype, amplitudes=amplitudes, offsets=offsets)
sim.noise_adder = SnrNoiseAdder(seed=sim.seed, snr=snr)

rand_rots = sim.rots

integrator = IcosahedronIntegrator(ell_max=4)

print("ell = {}".format(integrator.ell))

eye = np.eye(integrator.ell)
eye[:,0] = 1
weights = integrator.coeffs2weights(eye)
print(weights.shape)
integrals = np.sum(weights, axis=1)
print(integrals)
print(integrals.shape)

# M = integrator.b2w.T
#
# PI = M.T @ np.linalg.inv(M @ M.T)
#
# weights = np.eye(60)
#
# betas = (PI@weights).T

# sim.rots = integrator.rots

images = sim.images(0, num_imgs, enable_noise=True)


problem = LiftingProblem(imgs=images,
                         vol=vols,
                         filter=filters[0],
                         integrator=integrator,
                         # rots_prior=rand_rots,
                         )

# problem.rots_dcoef = betas

# print("weight sum = {}".format(np.sum(problem.integrator.coeffs2weights(problem.rots_dcoef) ** 2, axis=0)))
#
# Im = problem.vol_forward(problem.vol)
# print(Im)
# print(problem.dens_adjoint_forward(Im))
#
# basis = FBBasis3D((img_size, img_size, img_size))
#
# vol_update = LeastSquaresCGUpdate(problem, basis)
# vol = vol_update.update(tol=0.75)
#
# mean_estimator_up = MeanEstimator(sim, basis)
# mean_est_up = mean_estimator_up.estimate(tol=0.75)
#
# diff = vol - mean_est_up
#
# print(max(np.abs(diff.asnumpy())))

dens0 = problem.rots_dcoef[0]
print("before update dens = {}".format(dens0))

weights = problem.integrator.coeffs2weights(dens0)
print("weights = {}".format(weights))
print("summed weights = {}".format(np.sum(weights)))

quadratic_optimisation_update(problem=problem, sq_sigma=1e-10, reg1=0.001)
# betas = dens_update.update()

dens0_up = problem.rots_dcoef[0]
print("before after dens = {}".format(dens0_up))

weights = problem.integrator.coeffs2weights(dens0_up)
print("weights = {}".format(weights))
print("summed weights = {}".format(np.sum(weights)))