import itertools
import numpy as np
import os
import quaternionic

from scipy.spatial import ConvexHull

from solvers.lifting.integration.discretized_manifolds import DiscretizedManifold
from solvers.lifting.integration.discretized_manifolds.tools import normalize  #, quaternion_so3

class SO3(DiscretizedManifold):
    """ 3-dimensional rotational group represented using unit quaternions """
    ndim = 3
    # nembdim = 9

    def __init__(self, quats=None, h=None):
        """ Setup a simplicial grid on SO(3).
        Args:
            h : maximal length of edges in the triangulation
        """
        self.verts, self.simplices = so3mesh_initialization(quats)
        DiscretizedManifold.__init__(self, h)

    def mesh(self, h):
        triverts = self.verts[self.simplices]
        maxedgelen = self.dist(triverts, triverts).max()
        if h is not None:
            rep = max(0, np.ceil(np.log2(maxedgelen/h)))
        else:
            rep = 0
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
        if repeat == 0: return verts, simplices

        nverts = verts.shape[0]
        nsimplices = simplices.shape[0]

        edges = []
        edgecenters = np.zeros((nverts, nverts), dtype=np.int64)
        for j,sim in enumerate(simplices):
            for e in itertools.combinations(sim, 2):
                if edgecenters[e[0],e[1]] == 0:
                    edges.append(e)
                    edgecenters[e[0],e[1]] = edgecenters[e[1],e[0]] = nverts
                    nverts += 1
        edges = np.array(edges, dtype=np.int64)
        barycenters = np.int64(np.arange(nverts, nverts + nsimplices))
        nverts += nsimplices
        verts = np.concatenate((verts,
            self.mean(verts[edges][None], np.ones((1,1,2)))[0,:,0],
            self.mean(verts[simplices][None], np.ones((1,1,4)))[0,:,0],), axis=0)

        newsims = []
        for j,sim in enumerate(simplices):
            p12 = edgecenters[sim[0],sim[1]]
            p23 = edgecenters[sim[1],sim[2]]
            p13 = edgecenters[sim[0],sim[2]]
            p14 = edgecenters[sim[0],sim[3]]
            p24 = edgecenters[sim[1],sim[3]]
            p34 = edgecenters[sim[2],sim[3]]
            pc = barycenters[j]
            assert np.all([p12, p23, p13, p14, p24, p34, pc])
            newsims.extend([
                [sim[0], p12, p13, p14],
                [    pc, p12, p13, p14],
                [sim[1], p12, p23, p24],
                [    pc, p12, p23, p24],
                [sim[2], p13, p23, p34],
                [    pc, p13, p23, p34],
                [sim[3], p14, p24, p34],
                [    pc, p14, p24, p34],
            ])
        simplices = np.asarray(newsims)

        return self.mesh_refine(verts, simplices, repeat=repeat-1)

    # TODO HIER GEBLEVEN 7-7-21
    # TODO HIERNA: NIEUWE SO3 AAN DE REFINEMENT HANGEN EN OOK TESTEN WAT GEODESICS DOEN
    # TODO MAKE GEODESIC FUNCTION

    def _log(self, plocation, qfrom, out):
        """ log_p(q) = p log(sign(<p,q>) p^{-1} q)"""
        p = quaternionic.array(plocation).normalized
        q = quaternionic.array(qfrom).normalized

        pq = np.clip(np.einsum('ilm,ikm->ikl', q.ndarray, p.ndarray), -1.0, 1.0)  # TODO check what was actually going on here
        sign_pq = np.sign(pq[:,:,:,None])
        sign_pq[sign_pq == 0] = 1

        logpinvq = np.log(quaternionic.array(sign_pq * (np.reciprocal(p[:,:,None,:]) * q[:,None,:,:]).ndarray))
        logpinvq = 0.5 * (logpinvq - np.conj(logpinvq))  # make sure we are in the Lie algebra

        out[:] = (2 * p[:, :, None, :] * logpinvq).ndarray


    def _exp(self, plocation, vfrom, out):
        """ exp_p(v) = sign(w) * p exp(p^{-1} v) """
        p = quaternionic.array(plocation).normalized
        v = quaternionic.array(vfrom)  # TODO project onto tangent space

        pinvv = np.reciprocal(p)[:,:,None,:] * v[:,None,:,:]/2
        pinvv = 0.5 * (pinvv - np.conj(pinvv))

        q = (p[:,:,None,:] * np.exp(pinvv)).normalized
        sign_q = np.sign(q.ndarray[:, :, :, 0])
        sign_q[sign_q == 0] = 1

        out[:] = sign_q[:,:,:,None] * q.ndarray

    def _dist(self, x, y, out):
        np.einsum('ikm,ilm->ikl', x, y, out=out)
        out[:] = np.arccos(np.abs(np.clip(out, -1.0, 1.0)))  #2 * np.arccos(np.abs(np.clip(out, -1.0, 1.0)))
        # TODO use factor 2 here

    # def embed(self, x):
    #     return quaternion_so3(x).reshape(x.shape[:-1] + (9,))



def find_opverts(verts):
    inners = - verts @ verts.T  # n x n matrix
    opverts = np.argmax(inners, axis=0)

    # Check whether the indices are unique
    assert len(opverts) == len(np.unique(opverts))

    return opverts

def so3mesh_initialization(quats):
    """ 4-d base integrator where opposite points are
        identified (one of them is removed)
    Returns:
        verts : ndarray of floats
            Each row corresponds to a point on the 3-sphere.
        simplices : ndarray of floats
            Each row defines a simplex through indices into `verts`.
    """
    if quats is None:
        # Read quaternions from text file
        data_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "points"))
        filename = "sds031_03642.txt"
        # elif base_integrator == "icosahedron":
        #     filename = "sdr011_00120.txt"
        # else:
        #     raise NotImplementedError("This integrator is not available")
        filepath = os.path.join(data_dir, filename)
        verts = np.array_split(np.loadtxt(filepath), [4], axis=1)[0]  # used , dtype=self.dtype
    else:
        verts = np.vstack([quats, -quats])

    hull = ConvexHull(verts)  # We can do this due to the geometry of S3
    simplices = hull.simplices
    normals = hull.equations[:, 0:4]
    opverts = find_opverts(verts)

    # throw out half of all simplices based on orientation of its normal
    # vector with respect to the fixed reference direction
    reference_dir = np.array([1.0, 1e-4, 1.1e-4, 1.5e-4])
    simplices = simplices[normals.dot(reference_dir) > 0,:]
    newinds = np.zeros((verts.shape[0],), dtype=np.int64)
    vertkeep = (verts.dot(reference_dir) > 0)
    vertdiscard = np.logical_not(vertkeep)
    newinds[vertkeep] = np.arange(int(verts.shape[0]/2))
    newinds[vertdiscard] = newinds[opverts[vertdiscard]]
    return verts[vertkeep], np.ascontiguousarray(newinds[simplices])


