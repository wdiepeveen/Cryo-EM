import time

import numpy as np
from matplotlib import pyplot as plt

from pykeops.numpy import LazyTensor
import pykeops.config

dtype = "float64"

N = 10000 if pykeops.config.gpu_available else 1000  # Number of samples

# Sampling locations:
x = np.random.rand(N, 1).astype(dtype)

# Some random-ish 1D signal:
b = (
    x
    + 0.5 * np.sin(6 * x)
    + 0.1 * np.sin(20 * x)
    + 0.05 * np.random.randn(N, 1).astype(dtype)
)

def gaussian_kernel(x, y, sigma=0.1):
    x_i = LazyTensor(x[:, None, :])  # (M, 1, 1)
    y_j = LazyTensor(y[None, :, :])  # (1, N, 1)
    D_ij = ((x_i - y_j) ** 2).sum(-1)  # (M, N) symbolic matrix of squared distances
    return (-D_ij / (2 * sigma ** 2)).exp()  # (M, N) symbolic Gaussian kernel matrix

alpha = 1.0  # Ridge regularization

start = time.time()

K_xx = gaussian_kernel(x, x)
a = K_xx.solve(b, alpha=alpha)

end = time.time()

print(
    "Time to perform an RBF interpolation with {:,} samples in 1D: {:.5f}s".format(
        N, end - start
    )
)

# Extrapolate on a uniform sample:
t = np.reshape(np.linspace(0, 1, 1001), [1001, 1]).astype(dtype)

K_tx = gaussian_kernel(t, x)
mean_t = K_tx @ a

# 1D plot:
plt.figure(figsize=(8, 6))

plt.scatter(x[:, 0], b[:, 0], s=100 / len(x))  # Noisy samples
plt.plot(t, mean_t, "r")

plt.axis([0, 1, 0, 1])
plt.tight_layout()