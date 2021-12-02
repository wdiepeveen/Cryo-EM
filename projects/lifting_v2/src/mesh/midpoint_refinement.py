import numpy as np
import itertools
import os

from scipy.spatial import ConvexHull

from projects.lifting_v2.src.manifolds.so3 import SO3
from projects.lifting_v2.src.mesh import Mesh


class Midpoint_Refinement(Mesh):

    def __init__(self, quats=None, h=None):
        """ Setup a simplicial grid on SO(3).
        Args:
            h : maximal length of edges in the triangulation
        """
        self.manifold = SO3()
        self.verts, self.simplices = self.so3mesh_initialization(quats)
        Mesh.__init__(self, h)

    def mesh(self, h):
        triverts = self.verts[self.simplices]
        maxedgelen = self.manifold.dist(triverts, triverts).max()
        print(maxedgelen)
        if h is not None:
            rep = max(0, np.ceil(np.log2(maxedgelen / h)))
            print(rep)
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
        for j, sim in enumerate(simplices):
            for e in itertools.combinations(sim, 2):
                if edgecenters[e[0], e[1]] == 0:
                    edges.append(e)
                    edgecenters[e[0], e[1]] = edgecenters[e[1], e[0]] = nverts
                    nverts += 1
        edges = np.array(edges, dtype=np.int64)
        verts = np.concatenate((verts,
                                self.manifold.mean(verts[edges][None], np.ones((1, 1, 2)))[0, :, 0],), axis=0)

        newsims = []
        for j, sim in enumerate(simplices):
            p12 = edgecenters[sim[0], sim[1]]
            p23 = edgecenters[sim[1], sim[2]]
            p13 = edgecenters[sim[0], sim[2]]
            p14 = edgecenters[sim[0], sim[3]]
            p24 = edgecenters[sim[1], sim[3]]
            p34 = edgecenters[sim[2], sim[3]]
            assert np.all([p12, p23, p13, p14, p24, p34])
            newsims.extend([
                [sim[0], p12, p13, p14],  # K1
                [sim[1], p12, p23, p24],  # K2
                [sim[2], p13, p23, p34],  # K3
                [sim[3], p12, p13, p14],  # K4
                [sim[3], p12, p23, p24],  # K5
                [sim[3], p13, p23, p34],  # K6
                [sim[3], p12, p13, p23],  # K7
            ])
        simplices = np.asarray(newsims)

        return self.mesh_refine(verts, simplices, repeat=repeat - 1)

    def find_opverts(self, verts):
        inners = - verts @ verts.T  # n x n matrix
        opverts = np.argmax(inners, axis=0)

        # Check whether the indices are unique
        assert len(opverts) == len(np.unique(opverts))

        return opverts

    def so3mesh_initialization(self, quats):
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
            data_dir = os.path.join("data", "points")
            # data_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "points"))
            filename = "sds031_03642.txt"
            filepath = os.path.join(data_dir, filename)
            verts = np.array_split(np.loadtxt(filepath), [4], axis=1)[0]  # used , dtype=self.dtype
        else:
            verts = np.vstack([quats, -quats])

        hull = ConvexHull(verts)  # We can do this due to the geometry of S3
        simplices = hull.simplices
        normals = hull.equations[:, 0:4]
        opverts = self.find_opverts(verts)

        # throw out half of all simplices based on orientation of its normal
        # vector with respect to the fixed reference direction
        reference_dir = np.array([1.0, 1e-4, 1.1e-4, 1.5e-4])
        simplices = simplices[normals.dot(reference_dir) > 0, :]
        newinds = np.zeros((verts.shape[0],), dtype=np.int64)
        vertkeep = (verts.dot(reference_dir) > 0)
        vertdiscard = np.logical_not(vertkeep)
        newinds[vertkeep] = np.arange(int(verts.shape[0] / 2))
        newinds[vertdiscard] = newinds[opverts[vertdiscard]]
        return verts[vertkeep], np.ascontiguousarray(newinds[simplices])
