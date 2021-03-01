import os
import logging

from experiments.primal_dual_volume_and_rotation_reconstruction.preprocessing import preprocessing
from experiments.primal_dual_volume_and_rotation_reconstruction.experiment import experiment
from experiments.primal_dual_volume_and_rotation_reconstruction.postprocessing import postprocessing

from tools.exp_tools import Exp

logger = logging.getLogger(__name__)


def run_experiment():

    exp = Exp()

    exp.begin(prefix="pd_vol_rot_reconstruction")
    exp.dbglevel(4)

    # Set experiment parameters
    img_size_init = 33
    img_size = 65  # was 129 in rotation estimation paper
    nums_imgs = [512]  # [100, 500]  #, 1000]
    snr_range = [1/2]  # [1/2, 1/4]  #, 1/8, 1/16, 1/32, 1/64, 1/128, 1/256]
    max_it = 5

    # Select which parts to run
    skip_preprocessing = True
    skip_main_experiment = False

    # Set data path
    data_dir = "data"
    data_filename = "clean70SRibosome_vol_65p.mrc"
    data_path = os.path.join(data_dir, data_filename)

    # Set results folder if skip_preprocessing
    results_folder = "results/pd_vol_rot_reconstruction_21-03-01_17-52-06"
    if skip_preprocessing:
        assert results_folder is not None
    else:
        results_folder = None

    # Preprocessing
    if not skip_preprocessing:
        logger.info("Start Preprocessing")
        for i, snr in enumerate(snr_range):
            for j, num_imgs in enumerate(nums_imgs):

                if i == 0 and j == 0:
                    save_gt = True
                else:
                    save_gt = False
                    logger.info("Preprocessing for SNR = {} and {} images".format(snr, img_size))

                preprocessing(exp=exp,
                              num_imgs=num_imgs,
                              snr=snr,
                              img_size_init=img_size_init,
                              img_size=img_size,
                              data_path=data_path,
                              save_gt=save_gt
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

    logger.info("Start Postprocessing")
    # Postprocessing
    postprocessing(exp=exp,
                   nums_imgs=nums_imgs,
                   snr_range=snr_range,
                   results_folder=results_folder
                   )
