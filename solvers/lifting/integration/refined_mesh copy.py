import numpy as np
import os

import itertools
import logging

import spherical

from scipy.sparse import csc_matrix
from scipy.spatial import ConvexHull

from solvers.lifting.integration import Integrator
from solvers.lifting.integration.sd1821 import SphDes1821Integrator
from solvers.lifting.integration.icosahedron import IcosahedronIntegrator

logger = logging.getLogger(__name__)


class RefinedMeshIntegrator(Integrator):

    def __init__(self,
                 ell_max=3,  # Sets the angular resolution
                 mesh_norm=0.570595,  # Maximal length of edges in the triangulation - is 0.285297 # 0.14621 on S3
                 mesh_norm2=None,  # Maximal length of edges in triangulation for the constraint set
                 base_integrator="spherical-design",
                 dtype=np.float32,
                 ):

        # Read quaternions from text file
        data_dir = os.path.join(os.path.dirname(__file__), "points")
        if base_integrator == "spherical-design":
            filename = "sds031_03642.txt"
            n0 = 1821
        elif base_integrator == "icosahedron":
            filename = "sdr011_00120.txt"
            n0 = 60
        else:
            raise NotImplementedError("This integrator is not available")

        # filename = "sds031_03642.txt"
        filepath = os.path.join(data_dir, filename)

        verts = np.array_split(np.loadtxt(filepath), [4], axis=1)[0]  # used , dtype=self.dtype
        hull = ConvexHull(verts)  # We can do this due to the geometry of S3
        simplices = hull.simplices
        normals = hull.equations[:, 0:4]
        opverts = self.find_opverts(verts)

        # throw out half of all simplices based on orientation of its normal
        # vector with respect to the fixed reference direction
        # TODO reconsider whether we shouldnt do this afterwards
        reference_dir = np.array([1.0, 1e-4, 1.1e-4, 1.5e-4])
        simplices = simplices[normals.dot(reference_dir) > 0, :]
        newinds = np.zeros((verts.shape[0],), dtype=np.int64)
        vertkeep = (verts.dot(reference_dir) > 0)
        vertdiscard = np.logical_not(vertkeep)
        newinds[vertkeep] = np.arange(n0)
        newinds[vertdiscard] = newinds[opverts[vertdiscard]]

        self.verts, self.simplices = verts[vertkeep], np.ascontiguousarray(newinds[simplices])

        # Refine meshes
        angular_resolution = mesh_norm2 or np.pi/ell_max
        assert mesh_norm <= angular_resolution

        # Constraints mesh
        cverts, csimplices = self.mesh(angular_resolution/2)  # factor 1/2 for conversion mesh norm SO3 to S3

        # Rots mesh
        rverts, rsimplices = self.mesh(mesh_norm/2)  # factor 1/2 for conversion mesh norm SO3 to S3

        n = rverts.shape[0]
        cn = cverts.shape[0]

        # Initialize with dummy n
        super().__init__(dtype=dtype, n=n, ell=int((2 * ell_max + 1) * (2 * ell_max + 2) * (2 * ell_max + 3) / 6),
                         t=np.inf)

        self.quaternions = rverts

        # Copmpute U for all the rotations
        print("Construct U matrix")
        U = np.zeros((self.n, self.ell), dtype=complex)
        wigner = spherical.Wigner(ell_max)
        quats = self.quaternions
        for i in range(self.n):
            if self.n > 5000 and i%5000==0:
                print("Computing U matrix | progress: {} %".format(np.round(i/self.n *100)))
            g = quats[i]
            U[i] = wigner.D(g)

        # # Check that U integrates correctly
        integrals = np.real(np.sum(U, axis=0))/self.n
        print(integrals.shape)
        print("integrals U = {}".format(integrals))
        print("l_inf error = {}".format(np.max(np.abs(integrals - np.eye(1, self.ell)[0]))))

        U = csc_matrix(U)

        # Compute V
        print("Construct V matrix")
        # V = np.zeros((self.ell, self.ell), dtype=complex)  # TODO concstruct as sparse matrix
        # V = csc_matrix((self.ell, self.ell), dtype=complex)
        Vrow = []
        Vcol = []
        Vdata = []
        for l in range(ell_max + 1):
            for m in range(-l, l + 1):
                for n in range(-l, l + 1):
                    if m==0 and n==0:
                        index = wigner.Dindex(l, m, n)
                        # V[index,index] = 1
                        Vrow.append(index)
                        Vcol.append(index)
                        Vdata.append(1)
                    elif m>0 or (m==0 and n>0):
                        index = wigner.Dindex(l, m, n)
                        # V[index, index] = 1
                        Vrow.append(index)
                        Vcol.append(index)
                        Vdata.append(1)

                        indexx = wigner.Dindex(l, -m, -n)
                        # V[index, indexx] = (-1)**(m-n) * 1j
                        Vrow.append(index)
                        Vcol.append(indexx)
                        Vdata.append((-1)**(m-n) * 1j)
                    elif m<0 or (m==0 and n<0):
                        index = wigner.Dindex(l, m, n)
                        # V[index, index] = -1j
                        Vrow.append(index)
                        Vcol.append(index)
                        Vdata.append(-1j)

                        indexx = wigner.Dindex(l, -m, -n)
                        # V[index, indexx] = (-1) ** (m - n)
                        Vrow.append(index)
                        Vcol.append(indexx)
                        Vdata.append((-1) ** (m - n))

        V = csc_matrix((Vdata, (Vrow, Vcol)), shape=(self.ell, self.ell), dtype=complex)

        # V = sparse.csc_matrix(V)

        print("Construct B2W matrix")
        UV = U@V
        UV /= self.n
        # print(np.imag(UV))
        self.b2w = np.real(UV).T.astype(self.dtype)  # TODO asarray?

    def coeffs2weights(self, coeffs, cap_weights=True):
        weights = coeffs@self.b2w
        if cap_weights:
            weights = np.maximum(0, weights)

        return weights.astype(self.dtype)

    def weights2coeffs(self, weights):
        coeffs = weights@self.b2w.T

        return coeffs.astype(self.dtype)

    def find_opverts(self, verts):
        inners = - verts @ verts.T  # n x n matrix
        opverts = np.argmax(inners, axis=0)

        # Check whether the indices are unique
        assert len(opverts) == len(np.unique(opverts))

        return opverts

    def mesh(self, h):
        triverts = self.verts[self.simplices]

        maxedgelen = self.dist(triverts, triverts).max()
        rep = max(0, np.ceil(np.log2(maxedgelen / h)))
        return self.mesh_refine(self.verts, self.simplices, repeat=rep)

    def mesh_refine(self, verts, simplices, repeat=1):
        """ Refine a triangulation of SO(3)
        The algorithm applies the following procedure to a given triangulation:
        Each tetrahedron is split into eight tetrahedra using the tetrahedron's edge
        centers and its barycenter as new vertices.
        Args:
            verts : ndarray of floats, shape (nverts,4)
            simplices : ndarray of floats, shape (ntris,4)
            repeat : int
                The refinement procedure is iterated `repeat` times.
                If `repeat` is 0, the input is returned unchanged.
        Returns:
            verts : ndarray of floats, shape (nverts,4)
            simplices : ndarray of floats, shape (ntris,4)
        """
        print("update: repeat = {} | verts = {} | simplices = {}".format(repeat, verts.shape[0], simplices.shape[0]))
        if repeat == 0: return verts, simplices

        nverts = verts.shape[0]
        nsimplices = simplices.shape[0]

        edges = []
        edgecenters = np.zeros((nverts, nverts), dtype=np.int64)
        for j, sim in enumerate(simplices):
            for e in itertools.combinations(sim, 2):
                if edgecenters[e[0], e[1]] == 0:
                    edges.append(e)
                    edgecenters[e[0], e[1]] = edgecenters[e[1], e[0]] = nverts
                    nverts += 1

        edges = np.array(edges, dtype=np.int64)
        barycenters = np.int64(np.arange(nverts, nverts + nsimplices))
        nverts += nsimplices
        verts = np.concatenate((verts,
                                self.mean(verts[edges][None], np.ones((1, 1, 2)))[0, :, 0],
                                self.mean(verts[simplices][None], np.ones((1, 1, 4)))[0, :, 0],), axis=0)
        newsims = []
        for j, sim in enumerate(simplices):
            p12 = edgecenters[sim[0], sim[1]]
            p23 = edgecenters[sim[1], sim[2]]
            p13 = edgecenters[sim[0], sim[2]]
            p14 = edgecenters[sim[0], sim[3]]
            p24 = edgecenters[sim[1], sim[3]]
            p34 = edgecenters[sim[2], sim[3]]
            pc = barycenters[j]
            assert np.all([p12, p23, p13, p14, p24, p34, pc])
            newsims.extend([
                [sim[0], p12, p13, p14],
                [pc, p12, p13, p14],
                [sim[1], p12, p23, p24],
                [pc, p12, p23, p24],
                [sim[2], p13, p23, p34],
                [pc, p13, p23, p34],
                [sim[3], p14, p24, p34],
                [pc, p14, p24, p34],
            ])

        simplices = np.asarray(newsims)
        return self.mesh_refine(verts, simplices, repeat=repeat - 1)

    def mean(self, points, weights, out=None):
        """ Calculate arithmetic (geodesic) means of points on the manifold.
        Args:
            points : ndarray of floats, shape (npoints, nintdim)
                The first axis can be omitted.
            weights : ndarray of floats, shape (npoints,)
                The first axis can be omitted.
        Returns:
            ndarray of floats, shape (nintdim,)
        """
        nbatch, npointsets, npoints, nintdim = points.shape
        nweights = weights.shape[1]
        if out is None:
            out = np.zeros((nbatch, npointsets, nweights, nintdim))

        max_iter = 200
        nbatch, npointsets, npoints, nintdim = points.shape
        nweights = weights.shape[1]

        if nbatch * npointsets * nweights * npoints > 100:
            # Log only if it will take a bit longer...
            logger.info(("Computing {N} means of {npoints} "
                          "points in at most {maxiter} steps...").format(
                N=nbatch * npointsets * nweights,
                npoints=npoints, maxiter=max_iter))

        for i in range(nbatch):
            out[i] = points[i, :, np.argmax(weights[i], axis=-1), :].transpose(1, 0, 2)

        out_flat = out.reshape((nbatch * npointsets, nweights, nintdim))
        out_flat2 = out.reshape((nbatch * npointsets * nweights, nintdim))
        tmean = out.copy()
        tmean_flat = tmean.reshape((nbatch * npointsets, nweights, nintdim))
        tmean_flat2 = tmean.reshape((nbatch * npointsets * nweights, nintdim))
        tpoints = np.zeros((nbatch, npointsets, nweights, npoints, nintdim))
        tpoints_flat = tpoints.reshape((nbatch * npointsets, nweights, npoints, nintdim))
        points_flat = points.reshape((nbatch * npointsets, npoints, nintdim))

        w_sum_inv = 1.0 / np.einsum('ikl->ik', weights)
        for _iter in range(max_iter):
            self.log(out_flat, points_flat, out=tpoints_flat)
            np.einsum('ikm,ilkmt->ilkt', weights, tpoints, out=tmean)
            tmean *= w_sum_inv[:, None, :, None]
            self.exp(out_flat, tmean_flat)
            # out_flat2[:] = self.exp(out_flat2, tmean_flat2)

        # TODO check whether out has been updated at all now

        return out

    def log(self, location, pfrom, out=None):
        """ Inverse exponential map at `location` evaluated at `pfrom`
        Args:
            location : ndarray of floats, shape (nintdim,)
            pfrom : ndarray of floats, shape (nintdim,)
        Returns:
            ndarray of floats, shape (nintdim,)
        """
        nbatch, nlocations, nintdim = location.shape
        npoints = pfrom.shape[1]
        if out is None:
            out = np.zeros((nbatch, nlocations, npoints, nintdim))

        """ exp_l^{-1}(p) = d(l,p)*(sign(<p,l>)p - |<p,l>|l)/|p - <p,l>l| """
        # pl : <p,l>
        # fc : d(l,p)/|p - <p,l>*l| = arccos(|pl|)/sqrt(1 - pl^2)
        # out : fc*(sign(pl)*p - |pl|*l)
        pl = np.clip(np.einsum('ilm,ikm->ikl', pfrom, location), -1.0, 1.0)
        sign_pl = np.sign(pl[:, :, :, None])
        sign_pl[sign_pl == 0] = 1
        pl = np.abs(pl[:, :, :, None])
        fc = np.arccos(pl) / np.fmax(np.spacing(1), np.sqrt(1 - pl ** 2))
        out[:] = fc * (sign_pl * pfrom[:, None, :, :] - pl * location[:, :, None, :])

        return out

    def exp(self, location, vfrom, out=None):
        """ Inverse exponential map at `location` evaluated at `vfrom`
        Args:
            location : ndarray of floats, shape (nintdim,)
            vfrom : ndarray of floats, shape (nintdim,)
        Returns:
            ndarray of floats, shape (nintdim,)
        """
        nbatch, nlocations, nintdim = location.shape
        nvectors = vfrom.shape[1]
        if out is None:
            out = np.zeros((nbatch, nlocations, nvectors, nintdim))

        """ exp_l(v) = cos(|v|) * l  + sin(|v|) * v/|v| """
        vn = np.sqrt(np.einsum('ikm,ikm->ik', vfrom, vfrom))
        vnm = np.fmax(np.spacing(1), vn[:, None, :, None])
        out[:] = np.cos(vn[:, None, :, None]) * location[:, :, None, :]
        out += np.sin(vn[:, None, :, None]) / vnm * vfrom[:, None, :, :]
        # normalizing prevents errors from accumulating
        out[:] = self.normalize(out)

        return out

    def dist(self, x, y, out=None):
        """ Compute geodesic distance of points `x` and `y` on the manifold
        Args:
            x : ndarray of floats, shape (nintdim,)
            y : ndarray of floats, shape (nintdim,)
        Returns:
            ndarray of floats, shape ()
        """
        nbatch, nx, nintdim = x.shape
        ny = y.shape[1]
        if out is None:
            out = np.zeros((nbatch, nx, ny))
        np.einsum('ikm,ilm->ikl', x, y, out=out)
        out[:] = np.arccos(np.abs(np.clip(out, -1.0, 1.0)))
        return out

    def normalize(self, u, p=2, thresh=0.0):
        """ Normalizes u along the last axis with norm p.
        If  |u| <= thresh, 0 is returned (this mimicks the sign function).
        """
        ndim = u.shape[-1]
        multi = u.shape if u.ndim > 1 else None
        u = u.reshape(1, ndim) if multi is None else u.reshape(-1, ndim)
        ns = np.linalg.norm(u, ord=p, axis=1)
        fact = np.zeros_like(ns)
        fact[ns > thresh] = 1.0 / ns[ns > thresh]
        out = fact[:, None] * u
        return out[0] if multi is None else out.reshape(multi)