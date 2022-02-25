import logging
import numpy as np
import matplotlib.pyplot as plt

from projects.lifting_v2.src.integrators.base.sd1821mrx import SD1821MRx
from projects.lifting_v2.src.manifolds.so3 import SO3

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
    # solver = solver_data["solver"]
    vol_gt = solver_data["vol_gt"]  # Volume 65L
    snr = solver_data["SNR"]
    images = solver_data["images"]
    num_imgs = images.asnumpy().shape[0]


    rots_gt = exp.open_npy(data_dir, "rots_gt")
    rots_init = exp.open_npy(data_dir, "rots_init")
    rots_est = exp.open_npy(data_dir, "rots_est")

    rots_coeffs = exp.open_npy(data_dir, "rots_coeffs")

    # Load results

    integrator = SD1821MRx(repeat=mr_repeat)  #, dtype=dtype)

    # Process Stage 1 data:

    postfix = "_SNR{}_N{}_r{}".format(int(1 / snr), num_imgs, mr_repeat)

    # clean_image = sim.images(0, 1, enable_noise=False)
    # exp.save_im("data_projection_clean", clean_image.asnumpy()[0])
    exp.save_mrc("data_vol_orig", vol_gt.asnumpy()[0].astype(np.float32))

    # Get noisy projecrtion images
    # images = solver.plan.images
    exp.save_im("data_projection_noisy_0" + postfix, images.asnumpy()[0])
    exp.save_im("data_projection_noisy_1" + postfix, images.asnumpy()[1])
    exp.save_im("data_projection_noisy_2" + postfix, images.asnumpy()[2])

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

    manifold = SO3()

    #  Histogram Wasserstein distances
    squared_dist = manifold.dist(integrator.quaternions[None, :, :], rots_gt[None, :, :]).squeeze() ** 2

    W2 = np.sqrt(np.sum(squared_dist * rots_coeffs, axis=0))
    # print("W2.shape = {}".format(W2.shape))
    plt.figure()
    plt.hist(180 / np.pi * W2, bins=num_bins, range=(0, hist_range))
    # plt.xlabel(r"$W_2(\mu_{\mathcal{X}}^*,\delta_{p^*})\; (\degree)$")
    exp.save_fig("W2" + postfix)
    plt.show()

    #  Histogram distance rots init and gt
    dist_init = manifold.dist(rots_init[:, None, :], rots_gt[:, None, :]).squeeze()
    # print("dist_init.shape ={}".format(dist_init.shape))
    plt.figure()
    plt.hist(180 / np.pi * dist_init, bins=num_bins, range=(0, hist_range))
    # plt.xlabel(r"$d_{\mathrm{SO}(3)}(p_{\mathcal{X}}^0,p^*)\; (\degree)$")
    exp.save_fig("distance_init" + postfix)
    plt.show()

    #  Histogram distance rots ref and gt
    dist_est = manifold.dist(rots_est[:, None, :], rots_gt[:, None, :]).squeeze()
    # print("dist_est.shape = {}".format(dist_est.shape))
    plt.figure()
    plt.hist(180 / np.pi * dist_est, bins=num_bins, range=(0, hist_range))
    # plt.xlabel(r"$d_{\mathrm{SO}(3)}(p_{\mathcal{X}}^*,p^*)\; (\degree)$")
    exp.save_fig("distance_est" + postfix)
    plt.show()

    # TODO table of the mean distances (3x) -> output in .txt file
