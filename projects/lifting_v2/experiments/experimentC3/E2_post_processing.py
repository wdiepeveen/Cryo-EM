import logging
import numpy as np
import matplotlib.pyplot as plt

from aspire.utils.coor_trans import (
    get_aligned_rotations,
    get_rots_mse,
    register_rotations,
)

from projects.lifting_v2.src.util.rots_container import RotsContainer

logger = logging.getLogger(__name__)


def post_processing(exp=None,
                    mr_repeat=None,
                    # Density settings
                    # Histogram settings
                    num_bins=100,
                    hist_drange=20,
                    hist_dvrange=200,
                    hist_dticks=11,
                    hist_Jrange=20,
                    hist_Jvrange=400,
                    hist_Jticks=11,
                    # Results dir
                    results_folder=None,
                    ):
    logger.info(
        "Postprocessing started"
    )

    if results_folder is not None:
        data_dir = results_folder  # TODO fix as in run_file
    else:
        data_dir = exp.results_folder

    # Get all results in correct format to postprocess

    solver_data = exp.open_pkl(data_dir, "solver_data_r{}".format(mr_repeat))
    # Load data
    solver = solver_data["solver"]
    vol_gt = solver_data["vol_gt"]  # Volume 65L
    voxel_size = solver_data["voxel_size"]
    rots_gt = solver_data["rots_gt"]
    vol_init = solver_data["vol_init"]  # Volume 65L
    snr = solver_data["SNR"]
    num_imgs = solver.plan.N
    # Load results
    # cost = solver.cost
    # volume_est = solver.plan.vol
    # rots_est = solver.plan.rots

    # Process Stage 1 data:

    postfix = "_SNR{}_N{}_r{}".format(int(1 / snr), num_imgs, mr_repeat)

    # clean_image = sim.images(0, 1, enable_noise=False)
    # exp.save_im("data_projection_clean", clean_image.asnumpy()[0])
    exp.save_mrc("data_vol_orig", vol_gt.asnumpy()[0].astype(np.float32), voxel_size=voxel_size)
    exp.save_mrc("data_vol_init", vol_init.asnumpy()[0].astype(np.float32), voxel_size=voxel_size)

    # Get noisy projecrtion images
    images = solver.plan.images
    exp.save_im("data_projection_noisy_0" + postfix, images.asnumpy()[0])
    exp.save_im("data_projection_noisy_1" + postfix, images.asnumpy()[1])
    exp.save_im("data_projection_noisy_2" + postfix, images.asnumpy()[2])

    n = solver.plan.n
    J0 = solver.plan.J0
    eta = solver.plan.eta
    LB_J = 3 ** (-3/5) * J0 * n ** ((2 - 3*eta)/5)
    UB_J = J0 * n ** ((2 - 3*eta)/5)

    mean_J_table = []
    std_J_table = []

    mean_dists = []
    std_dists = []
    root_mean_squared_dists = []
    for i in range(solver.plan.max_iter):
        # Plot weights on Euler angles
        rots_coeffs = solver.rots_coeffs_iterates[i]

        fig = plt.figure()
        ax = fig.gca(projection='3d')

        angles = solver.plan.integrator.angles
        x = angles[:, 0]
        y = angles[:, 1]
        z = angles[:, 2]
        c = rots_coeffs[:, 0]
        mask = (c >= 1 / (100 * len(x)))

        print("integrated (averaged) density = {}".format(np.sum(c)))

        img = ax.scatter(x[mask], y[mask], z[mask], c=c[mask], cmap=plt.cool(), vmin=0, vmax=1)  # , alpha=0.1)
        ax.set_xlabel("$\phi$")
        ax.set_ylabel("$\\theta$")
        ax.set_zlabel("$\psi$")  # , rotation=0)
        ax.set_xlim([-np.pi, np.pi])
        ax.set_ylim([0, np.pi])
        ax.set_zlim([-np.pi, np.pi])
        plt.colorbar(img, ax=[ax], location='left')

        exp.save_fig("weights_on_angles" + postfix + "_i{}".format(i+1))

        # Histogram number of non-zero coeffs
        J = np.count_nonzero(rots_coeffs, axis=0)
        plt.figure()
        plt.hist(J, bins=num_bins, range=(0, hist_Jrange))
        plt.xlabel("Number of non-zero coefficients")
        plt.ylim(0, hist_Jvrange)
        plt.ylabel("Frequency")
        plt.xticks(np.linspace(0, hist_Jrange, hist_Jticks))
        exp.save_fig("J" + postfix + "_i{}".format(i + 1), save_eps=True)
        plt.show()

        mean_J = np.mean(J)
        std_J = np.std(J)
        mean_J_table.append(np.round(mean_J, 3))  # append mean
        std_J_table.append(np.round(std_J, 3))  # append std
        print("Predicted number of non-zero coefficients is between {} and {}".format(LB_J, UB_J))
        print("On average {} non-zero coefficients | std = {}".format(mean_J, std_J))

        #  Histograms distance rots ref and gt

        # Get register rotations after performing global alignment
        Q_mat, flag = register_rotations(solver.rots_iterates[i], rots_gt.rots)
        regrot = RotsContainer(num_imgs, dtype=solver.plan.dtype)
        regrot.rots = get_aligned_rotations(solver.rots_iterates[i], Q_mat, flag)
        mse_reg_est = get_rots_mse(regrot.rots, rots_gt.rots)
        logger.info(
            f"MSE deviation of the {i+1}:th estimated GD-refined rotations using register_rotations : {mse_reg_est}"
        )

        dist_est = solver.plan.integrator.manifold.dist(regrot.quaternions[:, None, :],
                                                        rots_gt.quaternions[:, None, :]).squeeze()
        # print("dist_est.shape = {}".format(dist_est.shape))
        plt.figure()
        plt.hist(180 / np.pi * dist_est, bins=num_bins, range=(0, hist_drange))
        plt.xlabel(r"Error $(\degree)$")
        plt.ylim(0, hist_dvrange)
        plt.ylabel("Frequency")
        plt.xticks(np.linspace(0, hist_drange, hist_dticks))
        exp.save_fig("distance_est" + postfix + "_i{}".format(i+1), save_eps=True)
        plt.show()

        mean_dists.append(np.mean(180 / np.pi * dist_est))
        std_dists.append(np.std(180 / np.pi * dist_est))

        print("On average distance is {} degrees | std = {}".format(np.mean(180 / np.pi * dist_est), np.std(180 / np.pi * dist_est)))

        root_mean_squared_dists.append(180 / np.pi * np.sqrt(np.mean(dist_est**2)))

        vol = solver.vol_iterates[i]
        exp.save_mrc("result_vol" + postfix + "_i{}".format(i + 1), vol.asnumpy()[0].astype(np.float32),
                     voxel_size=voxel_size)

    plt.figure()
    plt.plot(np.arange(1, solver.plan.max_iter+1), mean_dists)
    plt.fill_between(np.arange(1, solver.plan.max_iter+1), np.array(mean_dists) - np.array(std_dists), np.array(mean_dists) + np.array(std_dists), alpha=0.5, edgecolor='#CC4F1B', facecolor='#FF9848')
    plt.xlabel("Iteration")
    plt.ylabel(r"Error $(\degree)$")
    plt.ylim([0, hist_drange])
    exp.save_fig("rots_error_progression" + postfix, save_eps=True)
    plt.show()

    plt.figure()
    plt.plot(np.arange(1, solver.plan.max_iter + 1), mean_J_table)
    plt.fill_between(np.arange(1, solver.plan.max_iter + 1), np.array(mean_J_table) - np.array(std_J_table),
                     np.array(mean_J_table) + np.array(std_J_table),
                     alpha=0.5, edgecolor='#CC4F1B', facecolor='#FF9848')
    plt.xlabel("Iteration")
    plt.ylabel("# Non-zero coefficients")
    plt.ylim([0, hist_drange])
    exp.save_fig("J_progression" + postfix, save_eps=True)
    plt.show()

    # plt.figure()
    # plt.plot(np.arange(1, solver.plan.max_iter + 1), root_mean_squared_dists)
    # plt.xlabel("Iteration")
    # plt.ylabel(r"Root Mean Squared Error $(\degree)$")
    # plt.ylim([0, hist_drange])
    # exp.save_fig("rots_RMSD_progression" + postfix, save_eps=True)
    # plt.show()

    # # Plot cost
    # num_its = len(cost)
    #
    # plt.figure()
    # plt.plot(np.arange(num_its), cost)
    # plt.yscale('linear')
    # exp.save_fig("result_cost" + postfix)
    # plt.show()
    # print("costs = {}".format(cost))


