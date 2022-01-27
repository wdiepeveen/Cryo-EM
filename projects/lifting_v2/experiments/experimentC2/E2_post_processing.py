import logging
import numpy as np
import matplotlib.pyplot as plt

from aspire.utils.coor_trans import (
    get_aligned_rotations,
    get_rots_mse,
    register_rotations,
)

from noise.noise import SnrNoiseAdder

# from solvers.lifting.functions.rot_converters import mat2angle

logger = logging.getLogger(__name__)


def post_processing(exp=None,
                   # num_imgs=None,
                   # snr=None,
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

    solver_data = exp.open_pkl(data_dir, "solver_data")
    # solver_data = exp.open_pkl(data_dir, "solver_data" + postfix)
    # Load data
    solver = solver_data["solver"]
    refinement_solver = solver_data["refinement_solver"]
    vol_gt = solver_data["vol_gt"]  # Volume 65L
    vol_init = solver_data["vol_init"]  # Volume 65L
    rots_gt = solver_data["rots_gt"]
    rots_init = solver_data["rots_init"]
    snr = solver_data["SNR"]
    num_imgs = solver.plan.N
    # Load results
    angles = solver.plan.integrator.angles
    images = solver.plan.images
    cost = solver.cost


    volume_est = refinement_solver.plan.vol
    rots_est = refinement_solver.plan.rots

    # Process Stage 1 data:

    postfix = "_{}SNR_{}N".format(int(1 / snr), num_imgs)

    # clean_image = sim.images(0, 1, enable_noise=False)
    # exp.save_im("data_projection_clean", clean_image.asnumpy()[0])
    exp.save_mrc("data_vol_orig", vol_gt.asnumpy()[0].astype(np.float32))
    exp.save_mrc("data_vol_init", vol_init.asnumpy()[0].astype(np.float32))

    # Get noisy projecrtion image
    exp.save_im("data_projection_noisy" + postfix, images.asnumpy()[0])

    # Get density plots
    for i in range(solver.plan.max_iter):
        rots_coeffs = solver.rots_coeffs_iterates[i]

        fig = plt.figure()
        ax = fig.gca(projection='3d')

        x = angles[:, 0]
        y = angles[:, 1]
        z = angles[:, 2]
        c = rots_coeffs[:, 0]  # * len(x)  # only first density for visualization
        mask = (c >= 1/(100 * len(x)))
        # mask = (c >= max(c) / 2)

        print("integrated (averaged) density = {}".format(np.sum(c)))

        img = ax.scatter(x[mask], y[mask], z[mask], c=c[mask], cmap=plt.cool())  #, alpha=0.1)
        ax.set_xlabel("$\phi$")
        ax.set_ylabel("$\\theta$")
        ax.set_zlabel("$\psi$")  #, rotation=0)
        ax.set_xlim([-np.pi, np.pi])
        ax.set_ylim([0, np.pi])
        ax.set_zlim([-np.pi, np.pi])
        plt.colorbar(img)

        exp.save_fig("density" + postfix + "_i{}".format(i+1))

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

    # Make plots
    num_its = len(cost)

    plt.figure()
    plt.plot(np.arange(num_its) , cost)
    plt.yscale('linear')
    exp.save_fig("result_cost" + postfix)
    plt.show()
    print("costs = {}".format(cost))

    # Save results
    for i in range(solver.plan.max_iter):
        vol = solver.vol_iterates[i]
        exp.save_mrc("result_vol" + postfix + "_i{}".format(i+1), vol.asnumpy()[0].astype(np.float32))
    # exp.save_mrc("result_refined_vol" + postfix, refined_volume_est.asnumpy()[0])

    # Process Stage 2 data:

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

    # Histograms
    rot_wise_mse_reg_init = [180 / (np.pi * np.sqrt(2)) * np.sqrt(get_rots_mse(regrot[i][None], rots_gt[i][None])) for i
                             in range(num_imgs)]
    rot_wise_mse_reg_est = [180 / (np.pi * np.sqrt(2)) * np.sqrt(get_rots_mse(regrot2[i][None], rots_gt[i][None])) for i
                            in range(num_imgs)]
    # Factor 2 corresponds to our choice of metric used throughout the work
    num_bins = 100
    range_ = (min(min(rot_wise_mse_reg_init), min(rot_wise_mse_reg_est)),
              max(max(rot_wise_mse_reg_init), max(rot_wise_mse_reg_est)))

    postfix = "_{}SNR_{}N".format(int(1 / snr), num_imgs)

    plt.hist(rot_wise_mse_reg_init, bins=num_bins, range=range_)
    plt.xlabel("Distance (degrees)")
    exp.save_fig("rot_wise_mse_init" + postfix)
    plt.show()

    plt.hist(rot_wise_mse_reg_est, bins=num_bins, range=range_)
    plt.xlabel("Distance (degrees)")
    exp.save_fig("rot_wise_mse_est" + postfix)
    plt.show()

    # Save results
    exp.save_mrc("result_vol" + postfix, volume_est.asnumpy()[0].astype(np.float32))
