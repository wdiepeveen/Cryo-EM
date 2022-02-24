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
                    hist_range=50,
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
    rots_gt = solver_data["rots_gt"]
    vol_init = solver_data["vol_init"]  # Volume 65L
    snr = solver_data["SNR"]
    num_imgs = solver.plan.N
    # Load results
    # cost = solver.cost
    volume_est = solver.plan.vol
    rots_est = solver.plan.rots

    # Process Stage 1 data:

    postfix = "_SNR{}_N{}_r{}".format(int(1 / snr), num_imgs, mr_repeat)

    # clean_image = sim.images(0, 1, enable_noise=False)
    # exp.save_im("data_projection_clean", clean_image.asnumpy()[0])
    exp.save_mrc("data_vol_orig", vol_gt.asnumpy()[0].astype(np.float32))
    exp.save_mrc("data_vol_init", vol_init.asnumpy()[0].astype(np.float32))

    # Get noisy projecrtion images
    images = solver.plan.images
    exp.save_im("data_projection_noisy_0" + postfix, images.asnumpy()[0])
    exp.save_im("data_projection_noisy_1" + postfix, images.asnumpy()[1])
    exp.save_im("data_projection_noisy_2" + postfix, images.asnumpy()[2])

    mean_dists = []
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

        #  TODO Histogram Wasserstein distances

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
        plt.hist(180 / np.pi * dist_est, bins=num_bins, range=(0, hist_range))
        plt.xlabel(r"$d_{\mathrm{SO}(3)}(p_{\mathcal{X}}^*,p^*)\; (\degree)$")
        exp.save_fig("distance_est" + postfix + "_i{}".format(i+1))
        plt.show()

        mean_dists.append(np.mean(180 / np.pi * dist_est))

        vol = solver.vol_iterates[i]
        exp.save_mrc("result_vol" + postfix + "_i{}".format(i + 1), vol.asnumpy()[0].astype(np.float32))

    plt.figure()
    plt.plot(np.arange(1, solver.plan.max_iter+1), mean_dists)
    plt.xlabel("iteration")
    plt.ylabel(r"$d_{\mathrm{SO}(3)}(p_{\mathcal{X}}^*,p^*)\; (\degree)$")
    plt.ylim([0, hist_range])
    exp.save_fig("rots_error_progression" + postfix)
    plt.show()

    # # Plot cost
    # num_its = len(cost)
    #
    # plt.figure()
    # plt.plot(np.arange(num_its), cost)
    # plt.yscale('linear')
    # exp.save_fig("result_cost" + postfix)
    # plt.show()
    # print("costs = {}".format(cost))


