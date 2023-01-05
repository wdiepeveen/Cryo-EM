import numpy as np
import matplotlib.pyplot as plt


def projection_simplex(V, z=1, axis=None):
    """
    Projection of x onto the simplex, scaled by z:
        P(x; z) = argmin_{y >= 0, sum(y) = z} ||y - x||^2
    z: float or array
        If array, len(z) must be compatible with V
    axis: None or int
        axis=None: project V by P(V.ravel(); z)
        axis=1: project each V[i] by P(V[i]; z[i])
        axis=0: project each V[:, j] by P(V[:, j]; z[j])
    # Function from: https://gist.github.com/mblondel/c99e575a5207c76a99d714e8c6e08e89
    # Author: Mathieu Blondel
    """
    if axis == 1:
        # print("len(V) = {}".format(len(V)))
        n_features = V.shape[1]
        U = np.sort(V, axis=1)[:, ::-1]
        # print("U = {}".format(U[0]))
        z = np.ones(len(V)) * z
        cssv = np.cumsum(U, axis=1) - z[:, np.newaxis]
        # print("cssv = {}".format(cssv[1]))
        ind = np.arange(n_features) + 1
        cond = U - cssv / ind > 0
        rho = np.count_nonzero(cond, axis=1)
        theta = cssv[np.arange(len(V)), rho - 1] / rho
        return np.maximum(V - theta[:, np.newaxis], 0)

    elif axis == 0:
        return projection_simplex(V.T, z, axis=1).T

    else:
        V = V.ravel().reshape(1, -1)
        return projection_simplex(V, z, axis=1).ravel()

def compute_gamma(V, J0, eta):
    J = int(J0 * len(V) ** ((2 - eta)/3))
    print(J)
    Vj = np.sort(V, axis=0)
    VJ = Vj[0:J]
    summed_VJ = np.sum(VJ, axis=0)
    gamma = 1 / 2 * J0 ** 3 * (len(V) / J) ** 2 * (
                Vj[J] - 1 / J * summed_VJ)
    # lambdas = self.plan.J * (self.plan.n ** self.plan.eta) * (Fj[self.plan.J] - 1 / self.plan.J * summed_FJ)
    return gamma + 1e-16

# plot x^4 - 3* x^2 - 1/2 *x + 3 and variations
x_min = -2
x_max = 2

num_steps = 500
x = np.linspace(x_min, x_max, num_steps)
fx = x ** 4 - 3 * x ** 2 - 1 / 2 * x + 4
# oscilating part
k1 = 50
k2 = 30
oscx = 1 / 2 * np.sin(k1 * x) * np.cos(k2 * x)
foscx = fx + oscx

plt.figure()
plt.plot(x, foscx)
plt.show()

# subsampled regime
num_steps = 100
x = np.linspace(x_min, x_max, num_steps)
global_minimiser = 1.2645
# polynomial part
fx = x ** 4 - 3 * x ** 2 - 1 / 2 * x + 4
# oscilating part
k1 = 50
k2 = 30
oscx = 1 / 2 * np.sin(k1 * x) * np.cos(k2 * x)
foscx = fx + oscx

# RELION weights
# non-osc
beta_RELION = np.exp(-fx) / np.exp(-fx).sum()
mean_RELION = (x * beta_RELION).sum()
# osc
beta_RELION_osc = np.exp(-foscx) / np.exp(-foscx).sum()
mean_RELION_osc = (x * beta_RELION_osc).sum()

# our weights
J0 = 20
eta = 1.99
# non-osc
gamma = compute_gamma(fx, J0, eta)
beta_ESL = projection_simplex(- num_steps ** eta / gamma * fx)
mean_ESL = (x * beta_ESL).sum()
print(mean_ESL)
# osc
gamma_osc = compute_gamma(foscx, J0, eta)
beta_ESL_osc = projection_simplex(- num_steps ** eta / gamma_osc * foscx)
mean_ESL_osc = (x * beta_ESL_osc).sum()
print(mean_ESL_osc)


# old approach
plt.figure()
plt.plot(x, fx)
plt.plot(x, beta_RELION / beta_RELION.max())
plt.scatter([mean_RELION], [0.], c="orange", marker="x")
plt.scatter([global_minimiser], [0.], c="red")
plt.show()

plt.figure()
plt.plot(x, foscx)
plt.plot(x, beta_RELION_osc / beta_RELION_osc.max())
# plt.plot(x[(0.8<x) * (1.6>x)], fx[(0.8<x) * (x<1.6)] - 1/2, c="red")
plt.scatter([mean_RELION_osc], [0.], c="orange", marker="x")
# plt.scatter([global_minimiser], [0.], c="red")
plt.show()

# without the minimisers
prop_cycle = plt.rcParams['axes.prop_cycle']
colors = prop_cycle.by_key()['color']
plt.figure()
plt.plot(x, foscx)
plt.plot(x, beta_RELION_osc / beta_RELION_osc.max())
# plt.plot(x, beta_RELION_osc / beta_RELION_osc.max())
plt.scatter([global_minimiser], [0.], c="red")
plt.scatter([x[np.argmin(foscx)]], [0.], c=colors[2], marker="x")
plt.show()

# new approach
plt.figure()
plt.plot(x, fx)
plt.plot(x, beta_ESL / beta_ESL.max())
plt.scatter([mean_ESL], [0.], c="orange", marker="x")
plt.scatter([global_minimiser], [0.], c="red")
plt.show()

# plt.figure()
# plt.plot(x, foscx)
# plt.plot(x, beta_ESL_osc / beta_ESL_osc.max())
# plt.plot(x, fx - 1/2, c="red")
# plt.scatter([mean_ESL_osc], [0.], c="orange", marker="x")
# plt.scatter([global_minimiser], [0.], c="red")
# plt.show()

plt.figure()
plt.plot(x, foscx)
plt.plot(x, beta_ESL_osc / beta_ESL_osc.max())
# plt.plot(x, fx - 1/2, c="red")
plt.plot(x[(0.8<x) * (1.6>x)], fx[(0.8<x) * (x<1.6)] - 1/2, c="red")
plt.scatter([global_minimiser], [0.], c="red")
plt.scatter([mean_ESL_osc], [0.], c="orange", marker="x")
plt.show()
