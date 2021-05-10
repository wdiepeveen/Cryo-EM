import os
import logging

from experiments.lifting.preprocessing import preprocessing
from experiments.lifting.experiment_on import experiment
from experiments.lifting.postprocessing import postprocessing

from tools.exp_tools import Exp

logger = logging.getLogger(__name__)


def run_experiment():

    exp = Exp()

    exp.begin(prefix="lifting")  #, postfix="no_smudge")
    exp.dbglevel(4)

    # Set experiment parameters
    img_size = 33  # was 65 before and was 129 in rotation estimation paper
    max_it = 2

    # Set experiment iterables
    snr_range = [1 / 2]  # [1/2, 1/4, 1/8, 1/16, 1/32, 1/64, 1/128, 1/256]
    nums_imgs = [512]  # [100, 500, 1000]
    init_vol_smudge = 2

    # Select which parts to run
    skip_preprocessing = True
    skip_main_experiment = False

    # Set data path
    data_dir = "data"
    data_filename = "clean70SRibosome_vol_65p.mrc"
    data_path = os.path.join(data_dir, data_filename)

    # Set results folder if skip_preprocessing
    results_folder = "results/lifting_21-04-21_11-09-45"  # "results/lifting_21-04-26_12-22-06"

    # Preprocessing
    if not skip_preprocessing:
        results_folder = None
        logger.info("Start Preprocessing")
        for i, snr in enumerate(snr_range):
            for j, num_imgs in enumerate(nums_imgs):

                logger.info("Preprocessing for SNR = {} and {} images".format(snr, img_size))

                preprocessing(exp=exp,
                              num_imgs=num_imgs,
                              snr=snr,
                              img_size=img_size,
                              init_vol_smudge=init_vol_smudge,
                              data_path=data_path
                              )

    # Experiments

    if not skip_main_experiment or not skip_preprocessing:
        logger.info("Start Experiment")
        for i, snr in enumerate(snr_range):
            for j, num_imgs in enumerate(nums_imgs):
                logger.info("Running Experiment for SNR = {} and {} images".format(snr, num_imgs))

                experiment(exp=exp,
                           num_imgs=num_imgs,
                           snr=snr,
                           max_it=max_it,
                           results_folder=results_folder
                           )

        results_folder = None

    logger.info("Start Postprocessing")
    # Postprocessing
    postprocessing(exp=exp,
                   nums_imgs=nums_imgs,
                   snr_range=snr_range,
                   results_folder=results_folder
                   )
