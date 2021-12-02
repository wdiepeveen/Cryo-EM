

class Mesh:

    verts = None
    simplices = None

    def __init__(self, h):
        """ h : maximal length of edges in the triangulation """
        self.verts, self.simplices = self.mesh(h)
        self.nverts = self.verts.shape[0]
        self.nsimplices = self.simplices.shape[0]

    def mesh(self, h):
        raise NotImplementedError(
            "Subclasses should implement this and return vertices and simplices"
        )
