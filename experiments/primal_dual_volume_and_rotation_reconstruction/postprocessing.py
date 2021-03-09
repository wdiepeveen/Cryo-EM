import logging
import numpy as np
import matplotlib.pyplot as plt

from aspire.utils.coor_trans import (
    get_aligned_rotations,
    get_rots_mse,
    register_rotations,
)

from noise.noise import SnrNoiseAdder

logger = logging.getLogger(__name__)


def postprocessing(exp=None,
                   nums_imgs=None,
                   snr_range=None,
                   results_folder=None
                   ):
    logger.info(
        "Postprocessing started"
    )

    if results_folder is not None:
        data_dir = results_folder
    else:
        data_dir = exp.results_folder

    vol_fsc = np.zeros((len(snr_range), len(nums_imgs)))
    # rots_manifold_mse = np.zeros((len(snr_range), len(nums_imgs)))
    rots_mse = np.zeros((len(snr_range), 2 * len(nums_imgs)))

    # Get all results in correct format to postprocess
    for i, snr in enumerate(snr_range):
        for j, num_imgs in enumerate(nums_imgs):

            postfix = "_{}snr_{}n".format(int(1 / snr), num_imgs)

            # logger.info("Opening results from folder {}".format(data_dir))

            solver_data = exp.open_pkl(data_dir, "solver_data" + postfix)
            # Load data
            sim = solver_data["sim"]  # Simulator
            sim.noise_adder = SnrNoiseAdder(seed=sim.seed,
                                            snr=snr)  # bug in ASPIRE so that we cannot pickle sim like this
            vol_init = solver_data["vol_init"]  # Volume 65L
            vol_gt = solver_data["vol_gt"]  # Volume 65L
            rots_init = solver_data["rots_init"]
            rots_gt = solver_data["rots_gt"]  # np.Array
            # Load results
            volume_est = solver_data["volume_est"]
            rots_est = solver_data["rots_est"]
            cost = solver_data["cost"]
            relerror_u = solver_data["relerror_u"]
            relerror_g = solver_data["relerror_g"]
            relerror_tot = solver_data["relerror_tot"]

            if i == 0 and j == 0:
                clean_image = sim.images(0, 1, enable_noise=False)
                exp.save_im("data_projection_clean", clean_image.asnumpy()[0])
                exp.save_mrc("data_vol_orig", vol_gt.asnumpy()[0])

            # Get noisy projecrtion image
            noisy_image = sim.images(0, 1, enable_noise=True)
            exp.save_im("data_projection_noisy" + postfix, noisy_image.asnumpy()[0])
            exp.save_mrc("result_vol_preprocessing_{}snr_{}n".format(int(1 / snr), num_imgs), vol_init.asnumpy())

            # TODO FSC for volume

            # Get register rotations after performing global alignment
            Q_mat, flag = register_rotations(rots_init, rots_gt)
            regrot = get_aligned_rotations(rots_init, Q_mat, flag)
            mse_reg_init = get_rots_mse(regrot, rots_gt)
            logger.info(
                f"MSE deviation of the estimated initial rotations using register_rotations : {mse_reg_init}"
            )
            rots_mse[i, 2 * j] = mse_reg_init

            # check error on the rotations
            # Get register rotations after performing global alignment
            Q_mat, flag = register_rotations(rots_est, rots_gt)
            regrot = get_aligned_rotations(rots_est, Q_mat, flag)
            mse_reg = get_rots_mse(regrot, rots_gt)
            # manifold_mse_reg = get_rots_manifold_mse(regrot, rots_gt)
            logger.info(
                f"MSE deviation of the estimated corrected rotations using register_rotations : {mse_reg}"
            )
            rots_mse[i, 2 * j + 1] = mse_reg

            # Make plots
            num_its = len(cost)

            plt.figure()
            plt.plot(np.arange(num_its) + 1, cost)
            plt.yscale('linear')
            exp.save_fig("result_cost" + postfix)

            plt.figure()
            plt.plot(np.arange(num_its) + 1, relerror_u)
            plt.yscale('log')
            exp.save_fig("result_relerror_u" + postfix)

            plt.figure()
            plt.plot(np.arange(num_its) + 1, relerror_g)
            plt.yscale('log')
            exp.save_fig("result_relerror_g" + postfix)

            plt.figure()
            plt.plot(np.arange(num_its) + 1, relerror_tot)
            plt.yscale('log')
            exp.save_fig("result_relerror" + postfix)

            # Save results
            exp.save_mrc("result_vol" + postfix, volume_est.asnumpy()[0])

    # Create tables
    rot_mse_headers = []
    for num_imgs in nums_imgs:
        rot_mse_headers.append("MSE N = {} (init)".format(num_imgs))
        rot_mse_headers.append("MSE N = {} (ours)".format(num_imgs))

    snr_str = ["SNR"]  # also header for the side-header column
    for snr in snr_range:
        snr_str.append("1/{}".format(int(1 / snr)))

    exp.save_table("result_table_rot_mse", rots_mse, headers=rot_mse_headers, side_headers=snr_str)
