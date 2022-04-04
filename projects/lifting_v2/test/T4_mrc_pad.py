import numpy as np
import os

import mrcfile

from projects.lifting_v2.data_generation.mrc_pad import mrc_pad

data_filename = "waving_spike_state22_88x76y88z"

mrc_pad(data_filename, 91)


# dtype = np.float32
#
# # Set data path
# data_dir = "data"
# data_filename = "waving_spike_state22_88x76y88z.mrc"
# data_path = os.path.join("..", "..", "..", data_dir, data_filename)
#
# infile = mrcfile.open(data_path)
# vol = infile.data.astype(dtype)
# L = 91
# print(vol.shape)
# xpad1 = int((L - vol.shape[0]) / 2)
# if (L - vol.shape[0]) % 2 == 0:
#     xpad2 = xpad1
# else:
#     xpad2 = int((L - vol.shape[0]) / 2) + 1
#
# ypad1 = int((L - vol.shape[1]) / 2)
# if (L - vol.shape[1]) % 2 == 0:
#     ypad2 = ypad1
# else:
#     ypad2 = int((L - vol.shape[1]) / 2) + 1
#
# zpad1 = int((L - vol.shape[2]) / 2)
# if (L - vol.shape[2]) % 2 == 0:
#     zpad2 = zpad1
# else:
#     zpad2 = int((L - vol.shape[2]) / 2) + 1
# # Pad vol
# padded_vol = np.pad(vol, [(xpad1, xpad2), (ypad1, ypad2), (zpad1, zpad2)], "constant")
#
# print(padded_vol.shape)
#
# print(padded_vol)
# vol_gt = Volume(infile.data.astype(dtype))


