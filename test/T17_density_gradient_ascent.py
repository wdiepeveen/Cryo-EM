import os
import numpy as np
import matplotlib.pyplot as plt
import spherical

from solvers.lifting.integration.hexacosichoron import HexacosichoronIntegrator
from solvers.lifting.integration.uniform import UniformIntegrator
from solvers.lifting.integration.sd1821 import SphDes1821Integrator
from solvers.lifting.functions.rot_converters import quat2angle

from T14v2_wigner_gradient import compute_grad


def compute_cost(integrator, quats, coeffs):
    wigner = spherical.Wigner(integrator.ell_max)
    D = wigner.D(quats)
    b2w = np.real(D @ integrator.V)
    cost = np.einsum("ij,ij->i", b2w, coeffs)
    return cost

def do_grad_step(integrator, quats, coeffs, stepsize):
    grad = compute_grad(integrator, quats, coeffs)
    print("quat = {}".format(quats[0]))
    print("grad = {}".format(grad[0]))
    print("|grad|^2= {}".format(grad[0].T @ grad[0]))
    return integrator.manifold.exp(quats, stepsize*grad)  # TODO there shouldn't be a minus here...

def gradient_ascent(integrator, start, coeffs, stepsize, max_iter=20):
    quats = start
    cost = compute_cost(integrator, quats, coeffs)
    print("cost = {}".format(cost[0]))
    costs = [cost]
    angles = quat2angle(quats)
    print("phi = {}, theta = {}, psi = {}".format(angles[0, 0], angles[0, 1], angles[0, 2]))
    for i in range(max_iter):
        print(i)
        # print(quats[0])
        quats = do_grad_step(integrator, quats, coeffs, stepsize)
        # Compute cost
        cost = compute_cost(integrator,quats,coeffs)
        print("cost = {}".format(cost[0]))
        costs.append(cost)
        angles = quat2angle(quats)
        print("phi = {}, theta = {}, psi = {}".format(angles[0, 0], angles[0, 1], angles[0, 2]))

    return quats, costs

if __name__ == '__main__':
    # Define initial betas (from experiment) save .npy file somewhere
    dir = os.path.join("..", "results")
    # folder = "lifting_21-06-29_14-37-07"  # hexicosachoron
    folder = "lifting_21-06-30_09-42-23"  # sd1821
    filename = "rotation_density_coeffs.npy"
    path = os.path.join(dir, folder, filename)

    coeffs = np.load(path)
    print("coeffs = {}".format(coeffs))

    # integrator = HexacosichoronIntegrator(ell_max=5)
    integrator = SphDes1821Integrator(ell_max=15)
    density = integrator.coeffs2weights(coeffs)

    # Plot density
    angles = integrator.angles

    x = angles[:, 0]
    y = angles[:, 1]
    z = angles[:, 2]
    c = density[0, :] * len(x)  # only first density for visualization
    print("Number of angles = {}".format(len(x)))

    fig = plt.figure()
    ax = fig.gca(projection='3d')

    img = ax.scatter(x, y, z, c=c, cmap=plt.cool())
    ax.set_xlabel("$\phi$")
    ax.set_ylabel("$\\theta$")
    ax.set_zlabel("$\psi$")  # , rotation=0)
    plt.colorbar(img)

    plt.show()

    # TODO start from highest likelihood and do gradient ascent

    print(np.argmax(density, axis=-1))
    print(np.argmax(density, axis=-1).shape)
    start = integrator.quaternions[np.argmax(density, axis=-1),:]
    print(start.shape)

    # print gradient at that point (is it non-zero?)
    gradient = compute_grad(integrator, start, coeffs)
    print(gradient)


    # TODO do gradient ascent and check outcome
    max_iter = 20
    sols, costs = gradient_ascent(integrator,start,coeffs,1e-6,max_iter=max_iter)
    print(sols)

    print(integrator.manifold.dist(start,sols))

    costs_ = np.zeros((density.shape[0],max_iter))
    for i in range(max_iter):
        print(i)
        costs_[:,i] = costs[i]

    plt.plot(range(max_iter), costs_[0])
    plt.show()
    # TODO check function value density at sols and compare to quats

    # TODO check manifold distance between initial point and final iterate
