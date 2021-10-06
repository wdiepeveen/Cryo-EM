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
                   num_imgs=None,
                   snr=None,
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

    postfix = "_{}SNR_{}N".format(int(1 / snr), num_imgs)

    # logger.info("Opening results from folder {}".format(data_dir))

    solver_data = exp.open_pkl(data_dir, "solver_data" + postfix)
    # Load data
    sim = solver_data["sim"]  # Simulator
    sim.noise_adder = SnrNoiseAdder(seed=sim.seed,
                                    snr=snr)  # bug in ASPIRE so that we cannot pickle sim like this
    # vol_init = solver_data["vol_init"]  # Volume 65L
    vol_gt = solver_data["vol_gt"]  # Volume 65L
    rots_gt = solver_data["rots_gt"]
    # Load results
    volume_est = solver_data["volume_est"]
    # refined_volume_est = solver_data["refined_volume_est"]
    # density_est = solver_data["density_est"]
    angles = solver_data["angles"]
    density_est = solver_data["density_on_angles"]
    # refined_rots = solver_data["refined_rots_est"]
    cost = solver_data["cost"]

    clean_image = sim.images(0, 1, enable_noise=False)
    exp.save_im("data_projection_clean", clean_image.asnumpy()[0])
    exp.save_mrc("data_vol_orig", vol_gt.asnumpy()[0])

    # Get noisy projecrtion image
    noisy_image = sim.images(0, 1, enable_noise=True)
    exp.save_im("data_projection_noisy" + postfix, noisy_image.asnumpy()[0])
    # exp.save_mrc("result_vol_preprocessing_{}SNR_{}N".format(int(1 / snr), num_imgs), vol_init.asnumpy())

    # Get density plot
    fig = plt.figure()
    ax = fig.gca(projection='3d')

    x = angles[:, 0]
    y = angles[:, 1]
    z = angles[:, 2]
    c = density_est[:,0] * len(x)  # only first density for visualization
    
    print("integrated (averaged) density = {}".format(np.sum(c)/len(x)))

    img = ax.scatter(x, y, z, c=c, cmap=plt.cool(), alpha=0.1)
    ax.set_xlabel("$\phi$")
    ax.set_ylabel("$\\theta$")

    # ax.zaxis.set_rotate_label(False)
    ax.set_zlabel("$\psi$")  #, rotation=0)
    plt.colorbar(img)

    exp.save_fig("density" + postfix)

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
    plt.plot(np.arange(num_its) + 1, cost)
    plt.yscale('linear')
    exp.save_fig("result_cost" + postfix)

    # Save results
    exp.save_mrc("result_vol" + postfix, volume_est.asnumpy()[0])
    # exp.save_mrc("result_refined_vol" + postfix, refined_volume_est.asnumpy()[0])
