import numpy as np
import quaternionic

from projects.rkhs_lifting.src.manifolds import Manifold


class SO3(Manifold):
    """ 3-dimensional rotational group represented using unit quaternions """
    ndim = 3

    def _log(self, plocation, qfrom, out):
        """ log_p(q) = p log(sign(<p,q>) p^{-1} q)"""
        p = quaternionic.array(np.ascontiguousarray(plocation)).normalized
        q = quaternionic.array(np.ascontiguousarray(qfrom)).normalized

        pq = np.clip(np.einsum('ilm,ikm->ikl', q.ndarray, p.ndarray), -1.0, 1.0)  # TODO check what was actually going on here
        sign_pq = np.sign(pq[:,:,:,None])
        sign_pq[sign_pq == 0] = 1

        logpinvq = np.log(quaternionic.array(sign_pq * (np.reciprocal(p[:,:,None,:]) * q[:,None,:,:]).ndarray))
        logpinvq = 0.5 * (logpinvq - np.conj(logpinvq))  # make sure we are in the Lie algebra

        out[:] = (2 * p[:, :, None, :] * logpinvq).ndarray


    def _exp(self, plocation, vfrom, out):
        """ exp_p(v) = sign(w) * p exp(p^{-1} v) """
        p = quaternionic.array(np.ascontiguousarray(plocation)).normalized
        v = quaternionic.array(np.ascontiguousarray(vfrom))  # TODO project onto tangent space

        pinvv = np.reciprocal(p)[:,:,None,:] * v[:,None,:,:]/2
        pinvv = 0.5 * (pinvv - np.conj(pinvv))

        q = (p[:,:,None,:] * np.exp(pinvv)).normalized
        sign_q = np.sign(q.ndarray[:, :, :, 0])
        sign_q[sign_q == 0] = 1

        out[:] = sign_q[:,:,:,None] * q.ndarray

    def _dist(self, x, y, out):
        x = quaternionic.array(np.ascontiguousarray(x)).normalized.ndarray
        y = quaternionic.array(np.ascontiguousarray(y)).normalized.ndarray
        # print("Enter _dist | shapes x,y = ({}, {})".format(x.shape,y.shape))
        np.einsum('ikm,ilm->ikl', x, y, out=out)
        out[:] = 2 * np.arccos(np.abs(np.clip(out, -1.0, 1.0)))  #2 * np.arccos(np.abs(np.clip(out, -1.0, 1.0)))
