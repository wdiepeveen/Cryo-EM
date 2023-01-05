import numpy as np
import matplotlib.pyplot as plt

C = 10
eta = 1.
N_max = 1000000

N = np.cumsum(np.ones(N_max))
rN = np.sqrt(C) * N ** (- (1 + eta)/3)
discr = np.abs(np.floor(2 * rN * (N+1)) - 2 * rN * N)

plt.figure()
plt.semilogy(N, discr)
plt.semilogy(N, rN)
plt.show()

# rNm1 = np.sqrt(C) * ((N-1))** (- (1 + eta)/3)
plt.figure()
# plt.semilogy(2 *rNm1 * N - 2 * rN *N)
plt.semilogy(2 *rN[:-1] * N[1:] - 2 * rN[1:] *N[1:])
plt.semilogy(N, 2 * rN)
plt.show()
