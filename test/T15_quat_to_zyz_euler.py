import numpy as np
import quaternionic

# TODO write script that makes sure that our quats correspond to the right euler angles

# TODO also give random quaternion and convert to matrix with scipy and

# TODO also check whether spherical uses the same convention
# - can be done by giving same euler angles and seeing what scipy does and what spherical does

from scipy.spatial.transform import Rotation as R

r = R.from_matrix([[1, 0, 0],
                   [0, 1, 0],
                   [0, 0, 1]])

print(r.as_quat())

# Now compare R and quaternionic

# TODO define some angles

angles = np.array([[0,0,0], [1,2,3]])
print(angles)

# TODO compute quaternions from angles with R
Rrots = R.from_euler("ZYZ", angles)
Rquats = np.roll(Rrots.as_quat(),1,axis=-1)
# quats = np.roll(self._points.as_quat().astype(self.dtype),1,axis=-1)
sign_s = np.sign(Rquats[:, 0])
sign_s[sign_s == 0] = 1
Rquats = sign_s[:, None] * Rquats

print(Rquats)

# TODO compute " " with quaternionic
print(angles)
Q = quaternionic.array.from_euler_angles(1,2,3)
print(Q)  # Does not work

# TODO compate quaternions