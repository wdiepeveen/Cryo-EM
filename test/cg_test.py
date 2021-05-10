"""
This script illustrates the estimation of orientation angles using synchronization
matrix and voting method, based on simulated data projected from a 3D CryoEM map.
"""

import logging
import os

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

from solvers.lifting.functions.cost_functions import l2_data_fidelity, l2_vol_prior, l2_dens_prior, integral_dens_prior
from solvers.lifting.integration.true_rots import TrueRotsIntegrator
from solvers.lifting.update.volume.cg import conjugate_gradient_update
from solvers.lifting.problems.inside_norm import LiftingProblem
from solvers.lifting.solver import LiftingSolver

logger = logging.getLogger(__name__)



exp = Exp()

exp.begin(prefix="cg_test")  #, postfix="no_smudge")
exp.dbglevel(4)

num_imgs = 64  # was 512
snr = 1/2
img_size = 33  # was 33
init_vol_smudge = 2.

# Set data path
data_dir = "../data"
data_filename = "clean70SRibosome_vol_65p.mrc"
data_path = os.path.join(data_dir, data_filename)


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
    RadialCTFFilter(pixel_size, voltage, defocus=defocus, Cs=2.0, alpha=0.1)
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

offsets = np.zeros((num_imgs, 2)).astype(dtype)
amplitudes = np.ones(num_imgs)
sim = Simulation(L=img_size, n=num_imgs, vols=exp_vol_gt,
                 offsets=offsets, unique_filters=filters, amplitudes=amplitudes, dtype=dtype)
sim.noise_adder = SnrNoiseAdder(seed=sim.seed, snr=snr)


logger.info("Get true rotation angles generated randomly by the simulation object.")
rots_gt = sim.rots

# Initialize an orientation estimation object and perform view angle estimation
logger.info("Estimate rotation angles using synchronization matrix and voting method.")
orient_est = CLSyncVoting(sim, n_theta=36)  # was 72
orient_est.estimate_rotations()
rots_est = orient_est.rotations


# Start reconstruction
# rec_img_size = 11  # was 33
# vol_gt_ds = vol_gt.downsample((rec_img_size,) * 3)
prob = Simulation(L=img_size, n=num_imgs, vols=exp_vol_gt,
                 offsets=offsets, unique_filters=filters, amplitudes=amplitudes, dtype=dtype)
prob.noise_adder = SnrNoiseAdder(seed=sim.seed, snr=snr)
prob.rots = rots_est

# Specify the normal FB basis method for expending the 2D images
basis = FBBasis3D((img_size, img_size, img_size))

# Estimate the mean. This uses conjugate gradient on the normal equations for
# the least-squares estimator of the mean volume. The mean volume is represented internally
# using the basis object, but the output is in the form of an
# L-by-L-by-L array.
mean_estimator = MeanEstimator(prob, basis)

Tol = 1e-1

mean_est = mean_estimator.estimate(tol=Tol)
vol_init = Volume(gaussian_filter(mean_est.asnumpy(), init_vol_smudge))


# vol_init = Volume(zoom(vol_init.asnumpy()[0], img_size/rec_img_size, order=1))

vol_reg = 0.
dens_reg1 = 0.001
dens_reg2 = 1.

def cost_function(problem):
    fidelity_penalty = l2_data_fidelity(problem) *1e10

    vol_l2_penalty = vol_reg * l2_vol_prior(problem)
    dens_l2_penalty = dens_reg1 * l2_dens_prior(problem)
    if problem.rots_prior_integrands is not None:
        dens_prior_penalty = dens_reg2 * integral_dens_prior(problem)
    else:
        dens_prior_penalty = 0.

    cost = fidelity_penalty + vol_l2_penalty + dens_l2_penalty + dens_prior_penalty
    logger.info(
        "data penalty = {} | vol_reg penalty = {} | dens_reg1 penalty = {} | dens_reg2 penalty = {}".format(
            fidelity_penalty,
            vol_l2_penalty,
            dens_l2_penalty,
            dens_prior_penalty)
        )
    return cost

# Compute with own CG

integrator = TrueRotsIntegrator(rots=rots_est)
# integrator = UniformIntegrator(ell_max=4, n=num_imgs)
# integrator = IcosahedronIntegrator(ell_max=3)

imgs = prob.images(0, np.inf)

problem = LiftingProblem(imgs=imgs,
                      vol=Volume(np.zeros(exp_vol_gt.asnumpy()[0].shape)),  # this shouldn't be used in CG
                      filter=sim.unique_filters[0],
                      integrator=integrator,
                      rots_prior=None  # rots_init,
                      )

fvol = conjugate_gradient_update(problem, basis, tol=Tol, maxiter=18)

vol_cg1 = problem.vol
print("vol_cg1 cost = {}".format(cost_function(problem)))


postfix = "_{}SNR_{}N".format(int(1 / snr), num_imgs)

exp.save_mrc("result_vol_ASPIRE" + postfix, mean_est.asnumpy()[0])

exp.save_mrc("result_vol_CG1" + postfix, vol_cg1.asnumpy()[0])

# Test 1: Continue CG from conjugate_gradient_update fvol output

conjugate_gradient_update(problem, basis, x0=fvol, tol=Tol**2, maxiter=18)

vol_cg2 = problem.vol
print("vol_cg2 cost = {}".format(cost_function(problem)))

exp.save_mrc("result_vol_CG2" + postfix, vol_cg2.asnumpy()[0])

# Test 2: Continue CG from vol_init

fvol_init = basis.evaluate_t(vol_init.asnumpy()[0].T)

conjugate_gradient_update(problem, basis, x0=fvol_init, tol=Tol**2, maxiter=18)

vol_cg3 = problem.vol
print("vol_cg3 cost = {}".format(cost_function(problem)))

exp.save_mrc("result_vol_CG3" + postfix, vol_cg3.asnumpy()[0])


# basis = FBBasis3D((prob.L, prob.L, prob.L))

# fvol_init = basis.evaluate_t(vol_init.asnumpy().T)

def vol_update(problem, x0=fvol_init):
    return conjugate_gradient_update(problem, basis, x0=x0, regularizer=vol_reg, tol=Tol**2, maxiter=10)

def dens_update(problem):
    # quadratic_optimisation_update(problem, sq_sigma=sq_sigma, reg1=dens_reg1, reg2=dens_reg2)
    problem.rots_dcoef = np.eye(problem.ell, problem.n, dtype=problem.dtype)

max_it = 1

solver = LiftingSolver(problem=problem,
                       cost=cost_function,
                       max_it=max_it,
                       tol=Tol,
                       vol_update=vol_update,
                       dens_update=dens_update,
                       )

solver.solve(return_result=False)

vol_cg4 = solver.problem.vol
print("vol_cg4 cost = {}".format(cost_function(solver.problem)))

exp.save_mrc("result_vol_CG4" + postfix, vol_cg4.asnumpy()[0])

max_it = 2

solver = LiftingSolver(problem=problem,
                       cost=cost_function,
                       max_it=max_it,
                       tol=Tol,
                       vol_update=vol_update,
                       dens_update=dens_update,
                       )

solver.solve(return_result=False)

vol_cg5 = solver.problem.vol
print("vol_cg5 cost = {}".format(cost_function(solver.problem)))

exp.save_mrc("result_vol_CG5" + postfix, vol_cg5.asnumpy()[0])

# Save real vol and save vol from real rots
# exp.save("simulation_data_{}snr_{}n".format(int(1/snr), num_imgs),
#          ("sim", sim),
#          ("vol_init", vol_init),  # (img_size,)*3
#          ("rots_init", rots_est),
#          ("vol_gt", exp_vol_gt),  # (img_size,)*3
#          ("rots_gt", rots_gt)
#          )
