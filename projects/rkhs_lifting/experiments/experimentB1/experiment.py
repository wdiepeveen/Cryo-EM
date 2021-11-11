import numpy as np
import os
import logging

from projects.rkhs_lifting.experiments.experimentB1.E1_run import run_experiment
from projects.rkhs_lifting.experiments.experimentB1.E2_post_processing import post_processing

from tools.exp_tools import Exp

logger = logging.getLogger(__name__)

# os.chdir("/Users/wdiepeveen/Documents/PhD/Projects/2 - Cryo-EM/src/src/Cryo-EM")  # TODO This is shit -> write parser
os.chdir(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))  # TODO see if this works for time being

# Set data path
data_dir = os.path.join("results","expA2_21-10-27_10-39-37_L65_N512_r2_k90_lamV10_lamD-10")
data_filename = "solver_data_2SNR_512N"
# data_path = os.path.join(data_dir, data_filename)

# Set results folder if skip_preprocessing
# results_folder = "results/lifting_21-04-21_11-09-45"  # "results/lifting_21-04-26_12-22-06"
# results_folder ="results/expB1_21-11-03_16-27-07_L65_N512_k90_l10_i8_lamV10"
# results_folder = "results/expB1_21-11-04_16-46-35_L65_N512_k90_l10_i8_lamV10_gdMaxIter10_gdStep8"
results_folder = "results/expB1_21-11-11_13-06-29_L65_N50_k90_l10_i8_lamV10_gdMaxIter0_gdStep8"

# Experiment parameters
img_size = 65  # was 65 before and was 129 in rotation estimation paper
snr = 1 / 2  # [1/2, 1/4, 1/8, 1/16, 1/32, 1/64, 1/128, 1/256]
num_imgs = 8  # 512 capped
k = 90
l = 10
kernel_radius = np.pi/k  # radius of kernel
i = 0.8
# integrator_radius = np.sqrt(2) *2 * kernel_radius / (l-1)  # i * kernel_radius
integrator_radius = i * kernel_radius
# print("i={}".format(np.sqrt(2) *2 / (l-1)))
volume_reg_param = 1e10
rots_batch_size = 8192

gd_step_size = 8
stop_rots_gd = 10

# Directly start experiment
exp = Exp()

exp.begin(prefix="expB1", postfix="L{}_N{}_k{}_l{}_i{}_lamV{}_gdMaxIter{}_gdStep{}".format(img_size,num_imgs,k,l,int(10*i),int(np.log(volume_reg_param)/np.log(10)),stop_rots_gd,gd_step_size))
exp.dbglevel(4)

# Experiment
skip_experiment = False

if not skip_experiment:
    logger.info("Start Experiment")
    logger.info("Running Experiment for SNR = {} and {} images".format(snr, num_imgs))

    run_experiment(exp=exp,
                   num_imgs=num_imgs,
                   snr=snr,
                   img_size=img_size,
                   kernel_radius=kernel_radius,
                   l=l,
                   integrator_radius=integrator_radius,
                   stop_rots_gd=stop_rots_gd,
                   gd_step_size=gd_step_size,
                   volume_reg_param=volume_reg_param,
                   data_dir=data_dir,
                   data_filename=data_filename,
                   )


logger.info("Start Postprocessing")
# Postprocessing
post_processing(exp=exp,
               num_imgs=num_imgs,
               snr=snr,
               # results_folder=results_folder #?
               )

