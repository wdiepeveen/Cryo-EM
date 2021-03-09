import logging
import os

import mrcfile
import numpy as np
import matplotlib.pyplot as plt

from aspire.image.xform import NoiseAdder
from aspire.operators import RadialCTFFilter,ScalarFilter,ZeroFilter
from aspire.source.simulation import Simulation

from aspire.volume import Volume

from noise.noise import SnrNoiseAdder
from tools.exp_tools import Exp

logger = logging.getLogger(__name__)

DATA_DIR = os.path.join(os.path.dirname(__file__), "../data/")
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "../results/")


logger.info(
    "This script illustrates images at different SNR levels"
)


# Define a precision for this experiment
dtype = np.float32

# Set the sizes of images
img_size = 65

# Set the total number of images generated from the 3D map
num_imgs= 512  # 128

exp = Exp()
exp.begin(prefix="SNR_test", postfix="{}p_{}n".format(img_size,num_imgs))

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

print(max(vols.asnumpy()))

# Create a simulation object with specified filters and the downsampled 3D map
logger.info("Use downsampled map to creat simulation object.")
# TODO 1. do sim without
# TODO other option: 1. leave sim as it is, 2. compute SNR for each image later, 3. add noise, 4. replace image
# TODO compute new sim: 1. leave this sim as it is, 2. copy sim
snr_level = 1/2
sim = Simulation(L=img_size, n=num_imgs, vols=vols, unique_filters=filters, dtype=dtype)

# TODO Checken welke SNR we eruit krijgen. Als dit niet is wat we invoeren, moeten we wat aand die powerfilter doen die in noise.py zit
sim.noise_adder = SnrNoiseAdder(seed=sim.seed, snr=snr_level)

print(sim.noise_adder is None)

# Compute SNR of different
im_var = np.var(sim.clean_images(0,np.inf).asnumpy(),axis=(1,2))
noise_var = np.var(sim.images(0,np.inf,enable_noise=True).asnumpy() - sim.clean_images(0,np.inf).asnumpy(),axis=(1,2))
snr = im_var/noise_var
print(snr)

# Show typical images from simulation
num_plts = 3
images = sim.images(0,num_plts,enable_noise=True)
f, axarr = plt.subplots(1,num_plts)
for i in range(num_plts):
    axarr[i].imshow(images.asnumpy()[i])

plt.show()
