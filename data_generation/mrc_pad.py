import numpy as np
import os

import mrcfile


def mrc_pad(filename, L):
    dtype = np.float32

    # Set data path
    data_dir = "data"
    data_path = os.path.join("../projects/lifting_v2", "..", "..", data_dir, filename + ".mrc")

    infile = mrcfile.open(data_path)
    vol = infile.data.astype(dtype)

    assert max(vol.shape) <= L
    xpad1 = int((L - vol.shape[0]) / 2)
    if (L - vol.shape[0]) % 2 == 0:
        xpad2 = xpad1
    else:
        xpad2 = int((L - vol.shape[0]) / 2) + 1

    ypad1 = int((L - vol.shape[1]) / 2)
    if (L - vol.shape[1]) % 2 == 0:
        ypad2 = ypad1
    else:
        ypad2 = int((L - vol.shape[1]) / 2) + 1

    zpad1 = int((L - vol.shape[2]) / 2)
    if (L - vol.shape[2]) % 2 == 0:
        zpad2 = zpad1
    else:
        zpad2 = int((L - vol.shape[2]) / 2) + 1

    # Pad volume
    padded_vol = np.pad(vol, [(xpad1, xpad2), (ypad1, ypad2), (zpad1, zpad2)], "constant")

    # Save padded volume
    with mrcfile.new(os.path.join("../projects/lifting_v2", "..", "..", data_dir, filename + "_{}p".format(L) + ".mrc"), overwrite=True) as mrc:
        mrc.set_data(padded_vol)
        mrc.voxel_size = infile.voxel_size
