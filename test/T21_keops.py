import numpy as np

M, N = 10000, 10000
x = np.random.rand(M,4)
y = np.random.rand(N,4)
# print(x[0])
# print(y[0])
# print(x[0].T @ y[0])
# print((x * y).sum(-1)[0])
#
# x_i = x[:, None, :]  # (M, 1, 2) numpy array
# y_j = y[None, :, :]  # (1, N, 2) numpy array
# # D_ij = ((x_i - y_j) ** 2)  # (M, N) array of squared distances |x_i-y_j|^2
# D_ij = ((x_i * y_j).sum(-1))
# print(D_ij)
# s_i = np.argmin(D_ij, axis=1)  # (M,)   array of integer indices
# print(s_i[:10])

from pykeops.numpy import LazyTensor as LazyTensor_np

x_i = LazyTensor_np(
    x[:, None, :]
)  # (M, 1, 2) KeOps LazyTensor, wrapped around the numpy array x
y_j = LazyTensor_np(
    y[None, :, :]
)  # (1, N, 2) KeOps LazyTensor, wrapped around the numpy array y

kappa = 3
D_ij = (np.pi/kappa - 2* (x_i.normalize() * y_j.normalize()).sum(-1).abs().acos()).step() * (kappa * (x_i.normalize() * y_j.normalize()).sum(-1).abs().acos()).cos()**2 # **Symbolic** (M, N) matrix of squared distances
print(D_ij)

# TODO initialize random quaternions (and also normalize them)
quats = np.array(LazyTensor_np(np.random.rand(M,4)[:, None, :]).normalize().variables[0])
print(quats.shape)

# Initialize kernel
from solvers.lifting.integration.rkhs.kernels.rescaled_cosine import Rescaled_Cosine_Kernel

kernel = Rescaled_Cosine_Kernel(quats, np.pi/8)

# Do matrix vector product
matvec= kernel.apply_kernel(np.eye(M,1).squeeze())
print(matvec.variables)