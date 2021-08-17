import numpy as np
import spherical
import matplotlib.pyplot as plt

from scipy.sparse import csc_matrix

from solvers.lifting.integration.icosahedron import IcosahedronIntegrator


def compute_a(l, m, n):
    if l == -1:
        return 0
    else:
        return l / ((l + 1) * (2 * l + 1)) * np.sqrt(((l + 1) ** 2 - m ** 2) * ((l + 1) ** 2 - n ** 2))


def compute_b(l, m, n):
    if l == 0:
        return 0
    else:
        return m*n/(l*(l+1))


def compute_c(l, m, n):
    if l == 0:
        return 0
    else:
        return (l+1)/(l*(2*l+1)) * np.sqrt((l**2 - m**2)*(l**2 - n**2))


def compute_J1(quats):
    assert quats.shape[1] == 4
    # print(quats)
    # Compute jacobian
    J1 = np.zeros((quats.shape[0], 3, 4))

    # Compute squared components
    sq_sv3 = quats[:, 0] ** 2 + quats[:, 3] ** 2
    # print(sq_sv3)
    sq_v1v2 = quats[:, 1] ** 2 + quats[:, 2] ** 2
    # print(sq_v1v2)

    # Fill matrices
    eps = 1e-8  # prevent division by zero
    J1[:, 0, 0] = - quats[:, 3] / (sq_sv3 + eps)
    J1[:, 0, 1] = - quats[:, 2] / (sq_v1v2 + eps)
    J1[:, 0, 2] = quats[:, 1] / (sq_v1v2 + eps)
    J1[:, 0, 3] = quats[:, 0] / (sq_sv3 + eps)

    # J1[:, 1, 1] = (4 * quats[:, 1] ) / np.sqrt(1 - (1 - 2 * sq_v1v2) ** 2 + eps)
    # J1[:, 1, 2] = (4 * quats[:, 2] ) / np.sqrt(1 - (1 - 2 * sq_v1v2) ** 2 + eps)

    # J1[:, 1, 0] = - 2*quats[:, 0] * np.sqrt(sq_v1v2/(sq_sv3 + eps))
    # J1[:, 1, 1] = 2*quats[:, 1] * np.sqrt(sq_sv3/(sq_v1v2 + eps))
    # J1[:, 1, 2] = 2*quats[:, 2] * np.sqrt(sq_sv3/(sq_v1v2 + eps))
    # J1[:, 1, 3] = - 2*quats[:, 3] * np.sqrt(sq_v1v2 / (sq_sv3 + eps))

    J1[:, 1, 0] = - 2 * quats[:, 0] / np.sqrt(1 - (sq_sv3 - sq_v1v2)**2 + eps)
    J1[:, 1, 1] = 2 * quats[:, 1] / np.sqrt(1 - (sq_sv3 - sq_v1v2)**2 + eps)
    J1[:, 1, 2] = 2 * quats[:, 2] / np.sqrt(1 - (sq_sv3 - sq_v1v2)**2 + eps)
    J1[:, 1, 3] = - 2 * quats[:, 3] / np.sqrt(1 - (sq_sv3 - sq_v1v2)**2 + eps)

    J1[:, 2, 0] = J1[:, 0, 0]
    J1[:, 2, 1] = - J1[:, 0, 1]
    J1[:, 2, 2] = - J1[:, 0, 2]
    J1[:, 2, 3] = J1[:, 0, 3]

    # print("J1 = {}".format(J1))
    return J1


def compute_J2(integrator, quats, coeffs):
    # Compute J2
    wigner = spherical.Wigner(integrator.ell_max + 1)
    Arow = []
    Acol = []
    Adata = []
    Brow = []
    Bcol = []
    Bdata = []
    Crow = []
    Ccol = []
    Cdata = []
    for l in range(integrator.ell_max+1):
        for m in range(-l, l + 1):
            for n in range(-l, l + 1):
                index = wigner.Dindex(l, m, n)

                Arow.append(index)
                Acol.append(index)
                Adata.append(-m * 1j)

                Brow.append(index)
                Bcol.append(index)
                Bdata.append(-n * 1j)

                # Case distinction for C
                # l+1, m,n (+)
                a = compute_a(l, m, n)
                a_index = wigner.Dindex(l + 1, m, n)
                Crow.append(a_index)
                Ccol.append(index)
                Cdata.append(a)

                # l, m,n (-)
                b = compute_b(l, m, n)
                Crow.append(index)
                Ccol.append(index)
                Cdata.append(-b)

                # l-1, m,n (-)
                if l - 1 >= abs(m) and l - 1 >= abs(n):
                    c = compute_c(l, m, n)
                    c_index = wigner.Dindex(l - 1, m, n)
                    Crow.append(c_index)
                    Ccol.append(index)
                    Cdata.append(-c)

    ll = integrator.ell_max + 1

    # Compute dD/dphi
    # if integrator.Lam1 is None: # else compute it and store

    wig0 = spherical.Wigner(integrator.ell_max)
    D0 = wig0.D(quats)

    Lam1 = csc_matrix((Adata, (Arow, Acol)), shape=(integrator.ell, integrator.ell), dtype=np.complex64)
    # print("Lam1 = {}".format(Lam1))
    # print("Adata = {}".format(Adata))
    b2gr1 = np.real(D0 @ Lam1 @ integrator.V)  # shape (N,ell)
    # print(b2gr1)
    dDdphi = np.einsum("ij,ij->i", b2gr1, coeffs)
    # print(dDdphi)

    # Compute dD/dpsi
    Lam3 = csc_matrix((Bdata, (Brow, Bcol)), shape=(integrator.ell, integrator.ell), dtype=np.complex64)
    # print("Lam3 = {}".format(Lam3))
    # print("Bdata = {}".format(Bdata))
    b2gr3 = np.real(D0 @ Lam3 @ integrator.V)  # shape (N,ell)
    # print(b2gr3)
    dDdpsi = np.einsum("ij,ij->i", b2gr3, coeffs)
    # print(dDdpsi)

    #
    D = wigner.D(quats)
    elll = int((2*ll+1)*(2*ll+2)*(2*ll+3)/6)
    Lam2 = csc_matrix((Cdata, (Crow, Ccol)), shape=(elll, integrator.ell), dtype=np.complex64)
    # print("Lam2 = {}".format(Lam2))
    # plt.spy(Lam2)
    # plt.show()
    # l,m,n = (1,0,0)
    # ind = wigner.Dindex(l,m,n)
    # ind_a = wigner.Dindex(1, 0, 0)
    # print("Cdata = {} at index {}".format(Lam2[ind,ind], ind))
    b2gr2 = np.real(D @ Lam2 @ integrator.V)  # shape (N,ell)
    # print(b2gr2)
    dDdtheta = np.einsum("ij,ij->i", b2gr2, coeffs)
    # print(dDdtheta)
    # Compute sin theta and correct
    sq_v1v2 = quats[:, 1] ** 2 + quats[:, 2] ** 2
    eps = 1e-8  # prevent division by zero
    sintheta = np.sqrt(1 - (1 - 2*sq_v1v2)**2 + eps)  # sin(arccos(x)) = sqrt(1 - x^2)
    dDdtheta_ = dDdtheta/sintheta
    # print("dDdtheta_ = {}".format(dDdtheta_))

    J2 = np.vstack([dDdphi, dDdtheta_, dDdpsi]).T
    # print(J2.shape)

    return J2


def compute_grad(integrator, quats, coeffs):
    J1 = compute_J1(quats)
    print("J1 = {}".format(J1[0]))
    J2 = compute_J2(integrator, quats, coeffs)
    print("J2 = {}".format(J2[0]))
    J = np.einsum("ik,ikl->il", J2, J1)
    pJ = np.einsum("ik,ik->i", quats, J)
    # print(np.einsum("ik,ik->i", quats, quats))
    grad = J - pJ[:, None] * quats
    return grad

if __name__ == '__main__':
    integrator = IcosahedronIntegrator(ell_max=2)

    # print(compute_a(1,1,0))
    # print(compute_a(0,1,0))
    # print(compute_a(5,1,-4))
    #
    # print(compute_b(1,1,0))
    # print(compute_b(0,1,0))
    # print(compute_b(5,1,-4))
    #
    # print(compute_c(1,1,0))
    # print(compute_c(0,1,0))
    # print(compute_c(5,1,-4))

    quats = integrator.quaternions
    J1 = compute_J1(quats)  # TODO we shouldnt need an integrator, but solely the quats

    coeffs = np.ones((quats.shape[0], integrator.ell))

    J2 = compute_J2(integrator, quats, coeffs)

    J = np.einsum("ik,ikl->il", J2, J1)

    # print("J = {}".format(J))

    # project
    pJ = np.einsum("ik,ik->i", quats, J)

    grad = J - pJ[:, None] * quats

    # print("grad = {}".format(grad))
    # print("max grad = {}".format(np.max(np.abs(grad))))



