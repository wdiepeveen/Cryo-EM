import logging
import numpy as np
import matplotlib.pyplot as plt

from aspire.utils.coor_trans import (
    get_aligned_rotations,
    get_rots_mse,
    register_rotations,
)

from pymanopt.manifolds import Rotations

from noise.noise import SnrNoiseAdder

# from solvers.lifting.functions.rot_converters import mat2angle

logger = logging.getLogger(__name__)


def post_processing(exp=None,
                    data_dir=None,
                    data_filename=None,
                   results_folder=None
                   ):
    logger.info(
        "Postprocessing started"
    )

    if results_folder is not None:
        data_dir = results_folder
    else:
        data_dir = exp.results_folder

    # vol_fsc = np.zeros((len(snr_range), len(nums_imgs)))

    # Get all results in correct format to postprocess

    # logger.info("Opening results from folder {}".format(data_dir))

    # solver_data = exp.open_pkl(data_dir, "solver_data" + postfix)
    solver_data = exp.open_pkl(data_dir, data_filename)
    # Load data
    solver = solver_data["solver"]
    # solver2 = solver_data["solver2"]
    vol_gt = solver_data["vol_gt"]  # Volume 65L
    rots_gt = solver_data["rots_gt"]
    vol_init = solver_data["vol_init"]  # Volume 65L
    rots_init = solver_data["rots_init"]  # Volume 65L
    num_imgs = solver.plan.N
    snr = solver_data["SNR"]
    # Load results
    volume_est = solver.plan.vol
    # volume_est_gt_rots = solver2.plan.vol
    rots_est = solver.plan.rots
    # rots_est_gt_rots = solver2.plan.rots

    exp.save_mrc("data_vol_orig", vol_gt.asnumpy()[0])
    exp.save_mrc("data_vol_init", vol_init.asnumpy()[0])

    # Get register rotations after performing global alignment before gradient descent
    Q_mat, flag = register_rotations(rots_init[:num_imgs], rots_gt[:num_imgs])
    regrot = get_aligned_rotations(rots_init[:num_imgs], Q_mat, flag)
    mse_reg_init = get_rots_mse(regrot, rots_gt[:num_imgs])
    logger.info(
        f"MSE deviation of the estimated initial rotations using register_rotations : {mse_reg_init}"
    )

    # Get register rotations after performing global alignment after gradient descent
    Q_mat2, flag2 = register_rotations(rots_est, rots_gt[:num_imgs])
    regrot2 = get_aligned_rotations(rots_est, Q_mat2, flag2)
    mse_reg_est = get_rots_mse(regrot2, rots_gt[:num_imgs])
    logger.info(
        f"MSE deviation of the estimated GD-refined rotations using register_rotations : {mse_reg_est}"
    )

    # # Get register rotations after performing global alignment for reference rotations
    # Q_mat3, flag3 = register_rotations(rots_est_gt_rots, rots_gt[:num_imgs])
    # regrot3 = get_aligned_rotations(rots_est_gt_rots, Q_mat3, flag3)
    # mse_reg_ref = get_rots_mse(regrot3, rots_gt[:num_imgs])
    # logger.info(
    #     f"MSE deviation of the estimated reference GT rotations using register_rotations : {mse_reg_ref}"
    # )

    # Histograms
    rot_wise_mse_reg_init = [180/(np.pi*np.sqrt(2))*np.sqrt(get_rots_mse(regrot[i][None], rots_gt[i][None])) for i in range(num_imgs)]
    rot_wise_mse_reg_est = [180/(np.pi*np.sqrt(2))*np.sqrt(get_rots_mse(regrot2[i][None], rots_gt[i][None])) for i in range(num_imgs)]
    # Factor 2 corresponds to our choice of metric used throughout the work
    num_bins = 100
    range_ = (min(min(rot_wise_mse_reg_init),min(rot_wise_mse_reg_est)), max(max(rot_wise_mse_reg_init),max(rot_wise_mse_reg_est)))

    postfix = "_{}SNR_{}N".format(int(1 / snr), num_imgs)

    plt.hist(rot_wise_mse_reg_init, bins=num_bins,range=range_)
    plt.xlabel("Distance (degrees)")
    exp.save_fig("rot_wise_mse_init" + postfix)
    plt.show()

    plt.hist(rot_wise_mse_reg_est, bins=num_bins,range=range_)
    plt.xlabel("Distance (degrees)")
    exp.save_fig("rot_wise_mse_est" + postfix)
    plt.show()

    # Get noisy projecrtion image
    # noisy_image = sim.images(0, 1, enable_noise=True)
    # exp.save_im("data_projection_noisy" + postfix, noisy_image.asnumpy()[0])
    # exp.save_mrc("result_vol_preprocessing_{}SNR_{}N".format(int(1 / snr), num_imgs), vol_init.asnumpy())

    # # Get density plot
    # fig = plt.figure()
    # ax = fig.gca(projection='3d')
    #
    # x = angles[:, 0]
    # y = angles[:, 1]
    # z = angles[:, 2]
    # c = density_est[:,0] * len(x)  # only first density for visualization
    # mask = (c >= 1/2)
    # # mask = (c >= max(c) / 2)
    #
    # print("integrated (averaged) density = {}".format(np.sum(c)/len(x)))
    #
    # img = ax.scatter(x[mask], y[mask], z[mask], c=c[mask], cmap=plt.cool())  #, alpha=0.1)
    # ax.set_xlabel("$\phi$")
    # ax.set_ylabel("$\\theta$")
    # ax.set_zlabel("$\psi$")  #, rotation=0)
    # ax.set_xlim([-np.pi, np.pi])
    # ax.set_ylim([0, np.pi])
    # ax.set_zlim([-np.pi, np.pi])
    # plt.colorbar(img)
    #
    # exp.save_fig("density" + postfix)

    # refined_angles = mat2angle(refined_rots)
    # gt_angles = mat2angle(rots_gt)

    # Plot refined rot
    # xx = [refined_angles[0, 0], gt_angles[0,0]]
    # yy = [refined_angles[0, 1], gt_angles[0,1]]
    # zz = [refined_angles[0, 2], gt_angles[0,2]]
    # # cc = np.hstack([c, 200])
    # # TODO plot true rot also here
    #
    # # Get density plot
    # fig = plt.figure()
    # ax = fig.gca(projection='3d')
    #
    # img = ax.scatter(xx, yy, zz)  #, c=cc, cmap=plt.cool())
    # ax.set_xlabel("$\phi$")
    # ax.set_ylabel("$\\theta$")
    # ax.set_zlabel("$\psi$")  # , rotation=0)
    # plt.colorbar(img)
    #
    # exp.save_fig("refined_density" + postfix)

    # # Make plots
    # num_its = len(cost)
    #
    # plt.figure()
    # plt.plot(np.arange(num_its) + 1, cost)
    # plt.yscale('linear')
    # exp.save_fig("result_cost" + postfix)

    # Save results
    exp.save_mrc("result_vol" + postfix, volume_est.asnumpy()[0])
    # exp.save_mrc("result_vol_from_gt_rots" + postfix, volume_est_gt_rots.asnumpy()[0])
    # exp.save_mrc("result_refined_vol" + postfix, refined_volume_est.asnumpy()[0])
