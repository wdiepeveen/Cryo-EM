import logging
import numpy as np
import matplotlib.pyplot as plt

from projects.lifting_v2.src.integrators.base.sd1821mrx import SD1821MRx
from projects.lifting_v2.src.manifolds.so3 import SO3

logger = logging.getLogger(__name__)


def post_processing(exp=None,
                    mr_repeat=None,
                    eta_range=None,
                    # Density settings
                    # Histogram settings
                    num_bins=100,
                    hist_drange=50,
                    hist_vrange=50,
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
    # solver = solver_data["solver"]
    vol_gt = solver_data["vol_gt"]  # Volume 65L
    vol_init = solver_data["vol_init"]
    voxel_size = solver_data["voxel_size"]
    snr = solver_data["SNR"]
    images = solver_data["images"]
    num_imgs = images.asnumpy().shape[0]

    rots_gt = exp.open_npy(data_dir, "rots_gt")
    rots_init = exp.open_npy(data_dir, "rots_init")

    # clean_image = sim.images(0, 1, enable_noise=False)
    # exp.save_im("data_projection_clean", clean_image.asnumpy()[0])
    exp.save_mrc("data_vol_orig", vol_gt.asnumpy()[0].astype(np.float32), voxel_size=voxel_size)
    exp.save_mrc("data_vol_init", vol_init.asnumpy()[0].astype(np.float32), voxel_size=voxel_size)

    # Get noisy projecrtion images
    # images = solver.plan.images
    exp.save_im("data_projection_noisy_0", images.asnumpy()[0])
    exp.save_im("data_projection_noisy_1", images.asnumpy()[1])
    exp.save_im("data_projection_noisy_2", images.asnumpy()[2])

    manifold = SO3()

    #  Histogram distance rots init and gt
    dist_init = manifold.dist(rots_init[:, None, :], rots_gt[:, None, :]).squeeze()
    # print("dist_init.shape ={}".format(dist_init.shape))
    plt.figure()
    plt.hist(180 / np.pi * dist_init, bins=num_bins, range=(0, hist_drange))
    # plt.xlabel(r"$d_{\mathrm{SO}(3)}(p_{\mathcal{X}}^0,p^*)\; (\degree)$")
    plt.xlabel(r"Error $(\degree)$")
    plt.ylim(0, hist_vrange)
    plt.ylabel("Frequency")
    exp.save_fig("distance_init_r{}".format(mr_repeat), save_eps=True)
    plt.show()

    mean_g0dists = 180 / np.pi * np.mean(dist_init)
    std_g0dists = 180 / np.pi * np.std(dist_init)

    g0dists_table = np.array(
        [np.round(mean_g0dists, 3), np.round(std_g0dists, 3)])

    exp.save_table("g0dists_r{}".format(mr_repeat), g0dists_table[None])

    J_table = []
    W2dists_table = []
    gdists_table = []

    integrator = SD1821MRx(repeat=mr_repeat)  # , dtype=dtype)
    #  Histogram Wasserstein distances
    squared_dist = np.zeros((integrator.n, num_imgs))
    rots_batch_size = 1024
    for start in range(0, integrator.n, rots_batch_size):
        all_idx = np.arange(start, min(start + rots_batch_size, integrator.n))
        squared_dist[all_idx] = manifold.dist(integrator.quaternions[None, all_idx, :],
                                              rots_gt[None, :, :]).squeeze() ** 2

        logger.info(
            "Computing distance squared for {} rotations and {} images at {}%".format(integrator.n, num_imgs,
                                                                                      int((all_idx[
                                                                                               -1] + 1) / integrator.n * 100)))

    for eta in eta_range:

        print("=========== ETA = {}/100 ===========".format(int(eta * 100)))

        rots_est = exp.open_npy(data_dir, "rots_est_eta{}".format(int(eta * 100)))
        rots_coeffs = exp.open_npy(data_dir, "rots_coeffs_eta{}".format(int(eta * 100)))

        # Load results



        # Process Stage 1 data:

        postfix = "_r{}_eta{}".format(mr_repeat, int(eta * 100))

        # Plot weights on Euler angles
        # rots_coeffs = solver.plan.rots_coeffs

        fig = plt.figure()
        ax = fig.gca(projection='3d')

        angles = integrator.angles
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

        exp.save_fig("weights_on_angles" + postfix)

        # Compute J
        J = np.count_nonzero(rots_coeffs, axis=0)
        mean_J = np.mean(J)
        std_J = np.std(J)
        J_table.append(np.round(mean_J, 3))  # append mean
        J_table.append(np.round(std_J, 3))  # append std
        print("On average {} non-zero coefficients | std = {}".format(mean_J, std_J))


        W2 = np.sqrt(np.sum(squared_dist * rots_coeffs, axis=0))
        # print("W2.shape = {}".format(W2.shape))
        plt.figure()
        plt.hist(180 / np.pi * W2, bins=num_bins, range=(0, hist_drange))
        # plt.xlabel(r"$W_2(\mu_{\mathcal{X}}^*,\delta_{p^*})\; (\degree)$")
        plt.xlabel(r"Error $(\degree)$")
        plt.ylim(0, hist_vrange)
        plt.ylabel("Frequency")
        exp.save_fig("W2" + postfix, save_eps=True)
        plt.show()

        #  Histogram distance rots ref and gt
        dist_est = manifold.dist(rots_est[:, None, :], rots_gt[:, None, :]).squeeze()
        # print("dist_est.shape = {}".format(dist_est.shape))
        plt.figure()
        plt.hist(180 / np.pi * dist_est, bins=num_bins, range=(0, hist_drange))
        # plt.xlabel(r"$d_{\mathrm{SO}(3)}(p_{\mathcal{X}}^*,p^*)\; (\degree)$")
        plt.xlabel(r"Error $(\degree)$")
        plt.ylim(0, hist_vrange)
        plt.ylabel("Frequency")
        exp.save_fig("distance_est" + postfix, save_eps=True)
        plt.show()

        W2dists_table.append(np.round(180 / np.pi * np.mean(W2), 3))
        W2dists_table.append(np.round(180 / np.pi * np.std(W2), 3))
        gdists_table.append(np.round(180 / np.pi * np.mean(dist_est), 3))
        gdists_table.append(np.round(180 / np.pi * np.std(dist_est), 3))

        # root_mean_squared_Wdists = 180 / np.pi * np.sqrt(np.mean(W2 ** 2))
        # root_mean_squared_g0dists = 180 / np.pi * np.sqrt(np.mean(dist_init ** 2))
        # root_mean_squared_gdists = 180 / np.pi * np.sqrt(np.mean(dist_est ** 2))


    exp.save_table("J_r{}".format(mr_repeat), np.array(J_table)[None])
    exp.save_table("W2_r{}".format(mr_repeat), np.array(W2dists_table)[None])
    exp.save_table("g_r{}".format(mr_repeat), np.array(gdists_table)[None])


    # TODO check ratio's of different etas

    scaling_theory = integrator.n ** ((eta_range[-1] - np.array(eta_range[0:-1]))/5)
    scaling_practice = np.array(W2dists_table)[::2][0:-1]/np.array(W2dists_table)[-2]

    exp.save_table("scaling_theory_r{}".format(mr_repeat), np.round(scaling_theory,3)[None])
    exp.save_table("scaling_practice_r{}".format(mr_repeat), np.round(scaling_practice, 3)[None])

    # table_entries = np.array(
    #     [np.round(mean_J,3), np.round(std_J,3), np.round(mean_Wdists,3), np.round(std_Wdists,3), np.round(mean_g0dists,3), np.round(std_g0dists,3), np.round(mean_gdists,3), np.round(std_gdists,3)])
    #
    # exp.save_table("dists" + postfix, table_entries[None])
    #
    # RMSEtable_entries = np.array(
    #     [mean_J, std_J, root_mean_squared_Wdists, root_mean_squared_g0dists, root_mean_squared_gdists])
    # exp.save_table("RMSE" + postfix, RMSEtable_entries[None])
