import os
import logging

from experiments.angular_resolution.experiment import experiment
from experiments.angular_resolution.postprocessing import postprocessing

from tools.exp_tools import Exp

logger = logging.getLogger(__name__)


def run_experiment():

    exp = Exp()

    exp.begin(prefix="angular-resolution")  #, postfix="no_smudge")
    exp.dbglevel(4)

    # Set experiment parameters
    img_size = 65  # was 65 before and was 129 in rotation estimation paper

    # Set experiment iterables
    snr = 1 / 2
    num_imgs = 512

    # Select which parts to run
    skip_main_experiment = False

    # Set data path
    data_dir = "data"
    data_filename = "clean70SRibosome_vol_65p.mrc"
    data_path = os.path.join(data_dir, data_filename)

    # Set results folder if skip_preprocessing
    results_folder = None  # "results/lifting_21-04-21_11-09-45"  # "results/lifting_21-04-26_12-22-06"

    # Experiments

    ell_max_range = [3, 6, 9, 12, 15, 18]

    if not skip_main_experiment:
        logger.info("Start Experiment")
        for ell_max in ell_max_range:
            logger.info("Running Experiment with ell_max = {} for SNR = {} and {} images".format(ell_max, snr, num_imgs))

            experiment(exp=exp,
                       ell_max=ell_max,
                       num_imgs=num_imgs,
                       snr=snr,
                       img_size=img_size,
                       data_path=data_path)

    logger.info("Start Postprocessing")
    # Postprocessing
    postprocessing(exp=exp,
                   ell_max_range=ell_max_range,
                   num_imgs=num_imgs,
                   snr=snr,
                   results_folder=results_folder
                   )
