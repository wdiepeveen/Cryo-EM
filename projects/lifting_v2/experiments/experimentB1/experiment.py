import numpy as np
import os
import logging

from projects.lifting_v2.experiments.experimentB1.E1_run import run_experiment
from projects.lifting_v2.experiments.experimentB1.E2_post_processing import post_processing

from tools.exp_tools import Exp

logger = logging.getLogger(__name__)

# os.chdir("/Users/wdiepeveen/Documents/PhD/Projects/2 - Cryo-EM/src/src/Cryo-EM")  # TODO This is shit -> write parser
os.chdir(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))  # TODO see if this works for time being

# Set data path
# data_dir = os.path.join("results","expA1_21-12-15_14-12-25_L65_N512_r1_k90_lamV10_lamD500")
# data_dir = os.path.join("results","expA1_21-12-15_14-38-45_L65_N512_r2_k90_lamV10_lamD500")
data_dir = os.path.join("results","expA1_21-12-16_10-45-12_L65_N512_r2_k90_lamV10_lamD500")  # SNR=1/16
data_filename = "solver_data_16SNR_512N"
# data_path = os.path.join(data_dir, data_filename)

# Set results folder if skip_preprocessing
# results_folder = "results/lifting_21-04-21_11-09-45"  # "results/lifting_21-04-26_12-22-06"
# results_folder ="results/expB1_21-11-03_16-27-07_L65_N512_k90_l10_i8_lamV10"
# results_folder = "results/expB1_21-11-04_16-46-35_L65_N512_k90_l10_i8_lamV10_gdMaxIter10_gdStep8"
# results_folder = "results/expA1_21-12-15_14-12-25_L65_N512_r1_k90_lamV10_lamD500"

volume_reg_param = 1e10
# rots_batch_size = 8192

# Directly start experiment
exp = Exp()

exp.begin(prefix="expB1", postfix="L65_N512_r2_k90_lamV10_lamD500")
exp.dbglevel(4)

# Experiment
skip_experiment = False

if not skip_experiment:
    logger.info("Start Experiment")
    # logger.info("Running Experiment for SNR = {} and {} images".format(snr, num_imgs))

    run_experiment(exp=exp,
                   volume_reg_param=volume_reg_param,
                   data_dir=data_dir,
                   data_filename=data_filename,
                   )


logger.info("Start Postprocessing")
# Postprocessing
post_processing(exp=exp,
                data_dir=data_dir,
                data_filename=data_filename,
                )

