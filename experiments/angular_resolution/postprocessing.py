import logging
import numpy as np
import matplotlib.pyplot as plt

from noise.noise import SnrNoiseAdder

logger = logging.getLogger(__name__)


def postprocessing(exp=None,
                   ell_max_range=None,
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

    # Get all results in correct format to postprocess
    for i, ell_max in enumerate(ell_max_range):
        postfix = "_{}ell_{}SNR_{}N".format(ell_max, int(1 / snr), num_imgs)

        # logger.info("Opening results from folder {}".format(data_dir))

        solver_data = exp.open_pkl(data_dir, "solver_data" + postfix)
        # Load data
        images = solver_data["images"]
        vol_gt = solver_data["vol_gt"]  # Volume 65L
        angles_gt = solver_data["angles_gt"]
        # Load results
        volume_est = solver_data["volume_est"]
        density_est = solver_data["density_est"]

        if i == 0:
            exp.save_im("data_projection_noisy", images.asnumpy()[0])
            exp.save_mrc("data_vol_gt", vol_gt.asnumpy()[0])

        # Get density plot
        fig = plt.figure()
        ax = fig.gca(projection='3d')

        x = angles_gt[:, 0]
        y = angles_gt[:, 1]
        z = angles_gt[:, 2]
        c = density_est[0, :] * len(x)  # only first density for visualization

        img = ax.scatter(x, y, z, c=c, cmap=plt.cool())
        ax.set_xlabel("$\phi$")
        ax.set_ylabel("$\\theta$")
        ax.set_zlabel("$\psi$")  #, rotation=0)
        plt.colorbar(img)

        exp.save_fig("density" + postfix)

        # Save volume esults
        exp.save_mrc("result_vol" + postfix, volume_est.asnumpy()[0])
