import os
import logging

from experiments.primal_dual_volume_and_rotation_reconstruction.preprocessing import preprocessing
from experiments.primal_dual_volume_and_rotation_reconstruction.experiment import experiment
from experiments.primal_dual_volume_and_rotation_reconstruction.postprocessing import postprocessing

from tools.exp_tools import Exp

logger = logging.getLogger(__name__)


def run_experiment():

    exp = Exp()

    exp.begin(prefix="primal_dual")  #, postfix="no_smudge")
    exp.dbglevel(4)

    # Set experiment parameters
    img_size = 65  # was 65 before and was 129 in rotation estimation paper
    max_it = 5

    # Set experiment iterables
    snr_range = [1 / 2]  # [1/2, 1/4, 1/8, 1/16, 1/32, 1/64, 1/128, 1/256]
    nums_imgs = [512]  # [100, 500, 1000]
    init_vol_smudge = 0

    # Select which parts to run
    skip_preprocessing = False
    skip_main_experiment = False

    # Set data path
    data_dir = "data"
    data_filename = "clean70SRibosome_vol_65p.mrc"
    data_path = os.path.join(data_dir, data_filename)

    # Set results folder if skip_preprocessing
    results_folder = "results/primal_dual_vol_rot_reconstruction_21-03-08_15-00-30"

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
