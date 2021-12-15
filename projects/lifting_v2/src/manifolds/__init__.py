import logging
import numpy as np

from projects.lifting_v2.src.manifolds.util import cached_property, broadcast_io


class Manifold(object):
    """ Base class describing a triangulated manifold of dimension `ndim` """

    ndim = None

    @broadcast_io(1,1)
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
        self._log(location, pfrom, out)
        return out

    @broadcast_io(1,1)
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
        self._exp(location, vfrom, out)
        return out

    @broadcast_io((2,1),1)
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
        self._mean(points, weights, out)
        return out

    def _mean(self, points, weights, out, max_iter=20):
        nbatch, npointsets, npoints, nintdim = points.shape
        nweights = weights.shape[1]

        if nbatch*npointsets*nweights*npoints > 100:
            # Log only if it will take a bit longer...
            logging.info(("Computing {N} means of {npoints} "
                  "points in at most {maxiter} steps...").format(
                    N=nbatch*npointsets*nweights,
                    npoints=npoints, maxiter=max_iter))

        for i in range(nbatch):
            out[i] = points[i,:,np.argmax(weights[i], axis=-1),:].transpose(1,0,2)

        out_flat = out.reshape((nbatch*npointsets, nweights, nintdim))
        out_flat2 = out.reshape((nbatch*npointsets*nweights, nintdim))
        tmean = out.copy()
        tmean_flat = tmean.reshape((nbatch*npointsets, nweights, nintdim))
        tmean_flat2 = tmean.reshape((nbatch*npointsets*nweights, nintdim))
        tpoints = np.zeros((nbatch, npointsets, nweights, npoints, nintdim))
        tpoints_flat = tpoints.reshape((nbatch*npointsets, nweights, npoints, nintdim))
        points_flat = points.reshape((nbatch*npointsets, npoints, nintdim))

        w_sum_inv = 1.0/np.einsum('ikl->ik', weights)
        for _iter in range(max_iter):
            self.log(out_flat, points_flat, out=tpoints_flat)
            np.einsum('ikm,ilkmt->ilkt', weights, tpoints, out=tmean)
            tmean *= w_sum_inv[:,None,:,None]
            out_flat2[:] = self.exp(out_flat2, tmean_flat2)

    @broadcast_io(1,0)
    def dist(self, x, y, out=None):
        """ Compute geodesic distance of points `x` and `y` on the manifold
        Args:
            x : ndarray of floats, shape (nintdim,)
            y : ndarray of floats, shape (nintdim,)
        Returns:
            ndarray of floats, shape ()
        """
        # print("Enter dist | shapes x,y = ({}, {})".format(x.shape,y.shape))
        nbatch, nx, nintdim = x.shape
        ny = y.shape[1]
        if out is None:
            out = np.zeros((nbatch, nx, ny))
        self._dist(x, y, out)
        return out
