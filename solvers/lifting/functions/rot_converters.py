import numpy as np
from scipy.spatial.transform import Rotation as R


def mat2quat(mats):
    # TODO fix quats convention
    points = R.from_matrix(mats)
    return points.as_quat()


def quat2mat(quats):
    # TODO fix quats convention
    points = R.from_quat(quats)
    return points.as_matrix()


def mat2angle(mats):
    points = R.from_matrix(mats)
    return points.as_euler("ZYZ")


def angle2mat(angles):
    points = R.from_euler("ZYZ", angles)
    return points.as_matrix()


def quat2angle(quats):
    values = np.roll(quats, -1, axis=-1)
    points = R.from_quat(values)
    return points.as_euler("ZYZ")


def angle2quat(angles):
    # TODO fix quat convention
    points = R.from_euler("ZYZ", angles)
    return points.as_quat()
