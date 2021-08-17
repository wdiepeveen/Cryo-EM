import logging
import numpy as np
import matplotlib.pyplot as plt

from aspire.utils.coor_trans import (
    get_aligned_rotations,
    get_rots_mse,
    register_rotations,
)

from noise.noise import SnrNoiseAdder

from solvers.lifting.functions.rot_converters import mat2angle

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

    # Get all results in correct format to postprocess
    for i, snr in enumerate(snr_range):
        for j, num_imgs in enumerate(nums_imgs):

            postfix = "_{}SNR_{}N".format(int(1 / snr), num_imgs)

            # logger.info("Opening results from folder {}".format(data_dir))

            solver_data = exp.open_pkl(data_dir, "solver_data" + postfix)
            # Load data
            sim = solver_data["sim"]  # Simulator
            sim.noise_adder = SnrNoiseAdder(seed=sim.seed,
                                            snr=snr)  # bug in ASPIRE so that we cannot pickle sim like this
            vol_init = solver_data["vol_init"]  # Volume 65L
            vol_gt = solver_data["vol_gt"]  # Volume 65L
            rots_gt = solver_data["rots_gt"]
            # Load results
            volume_est = solver_data["volume_est"]
            refined_volume_est = solver_data["refined_volume_est"]
            density_est = solver_data["density_est"]
            angles, density = density_est
            refined_rots = solver_data["refined_rots_est"]
            cost = solver_data["cost"]

            if i == 0 and j == 0:
                clean_image = sim.images(0, 1, enable_noise=False)
                exp.save_im("data_projection_clean", clean_image.asnumpy()[0])
                exp.save_mrc("data_vol_orig", vol_gt.asnumpy()[0])

            # Get noisy projecrtion image
            noisy_image = sim.images(0, 1, enable_noise=True)
            exp.save_im("data_projection_noisy" + postfix, noisy_image.asnumpy()[0])
            exp.save_mrc("result_vol_preprocessing_{}SNR_{}N".format(int(1 / snr), num_imgs), vol_init.asnumpy())

            # Get density plot
            fig = plt.figure()
            ax = fig.gca(projection='3d')

            x = angles[:, 0]
            y = angles[:, 1]
            z = angles[:, 2]
            c = density[0, :] * len(x)  # only first density for visualization

            img = ax.scatter(x, y, z, c=c, cmap=plt.cool())
            ax.set_xlabel("$\phi$")
            ax.set_ylabel("$\\theta$")

            # ax.zaxis.set_rotate_label(False)
            ax.set_zlabel("$\psi$")  #, rotation=0)
            plt.colorbar(img)

            exp.save_fig("density" + postfix)

            refined_angles = mat2angle(refined_rots)
            gt_angles = mat2angle(rots_gt)

            # Plot refined rot
            xx = [refined_angles[0, 0], gt_angles[0,0]]
            yy = [refined_angles[0, 1], gt_angles[0,1]]
            zz = [refined_angles[0, 2], gt_angles[0,2]]
            # cc = np.hstack([c, 200])
            # TODO plot true rot also here

            # Get density plot
            fig = plt.figure()
            ax = fig.gca(projection='3d')

            img = ax.scatter(xx, yy, zz)  #, c=cc, cmap=plt.cool())
            ax.set_xlabel("$\phi$")
            ax.set_ylabel("$\\theta$")
            ax.set_zlabel("$\psi$")  # , rotation=0)
            plt.colorbar(img)

            exp.save_fig("refined_density" + postfix)

            # fig = plt.figure()
            # ax = fig.add_subplot(111, projection='3d')
            #
            # x = angles[:, 0]
            # y = angles[:, 1]
            # z = angles[:, 2]
            # c = density[0, :]  # only first density for visualization
            #
            # n = len(x)
            # mask = c > 10. / n
            # img = ax.scatter(x, y, z, c=c, s=50 * mask + 10*(1-mask), cmap=plt.cool())
            # plt.xlabel("phi")
            # plt.ylabel("theta")
            # # plt.("phi")
            # fig.colorbar(img)
            #
            # exp.save_fig("density_threshold" + postfix)

            # TODO FSC for volume

            # # Get register rotations after performing global alignment
            # Q_mat, flag = register_rotations(rots_init, rots_gt)
            # regrot = get_aligned_rotations(rots_init, Q_mat, flag)
            # mse_reg_init = get_rots_mse(regrot, rots_gt)
            # logger.info(
            #     f"MSE deviation of the estimated initial rotations using register_rotations : {mse_reg_init}"
            # )
            # rots_mse[i, 2 * j] = mse_reg_init
            #
            # # check error on the rotations
            # # Get register rotations after performing global alignment
            # Q_mat, flag = register_rotations(rots_est, rots_gt)
            # regrot = get_aligned_rotations(rots_est, Q_mat, flag)
            # mse_reg = get_rots_mse(regrot, rots_gt)
            # # manifold_mse_reg = get_rots_manifold_mse(regrot, rots_gt)
            # logger.info(
            #     f"MSE deviation of the estimated corrected rotations using register_rotations : {mse_reg}"
            # )
            # rots_mse[i, 2 * j + 1] = mse_reg

            # Make plots
            num_its = len(cost)

            plt.figure()
            plt.plot(np.arange(num_its) + 1, cost)
            plt.yscale('linear')
            exp.save_fig("result_cost" + postfix)

            # plt.figure()
            # plt.plot(np.arange(num_its) + 1, relerror_u)
            # plt.yscale('log')
            # exp.save_fig("result_relerror_u" + postfix)
            #
            # plt.figure()
            # plt.plot(np.arange(num_its) + 1, relerror_g)
            # plt.yscale('log')
            # exp.save_fig("result_relerror_g" + postfix)
            #
            # plt.figure()
            # plt.plot(np.arange(num_its) + 1, relerror_tot)
            # plt.yscale('log')
            # exp.save_fig("result_relerror" + postfix)

            # Save results
            exp.save_mrc("result_vol" + postfix, volume_est.asnumpy()[0])
            exp.save_mrc("result_refined_vol" + postfix, refined_volume_est.asnumpy()[0])

    # # Create tables
    # rot_mse_headers = []
    # for num_imgs in nums_imgs:
    #     rot_mse_headers.append("MSE N = {} (init)".format(num_imgs))
    #     rot_mse_headers.append("MSE N = {} (ours)".format(num_imgs))
    #
    # snr_str = ["SNR"]  # also header for the side-header column
    # for snr in snr_range:
    #     snr_str.append("1/{}".format(int(1 / snr)))

    # exp.save_table("result_table_rot_mse", rots_mse, headers=rot_mse_headers, side_headers=snr_str)
