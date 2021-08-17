import numpy as np

def proj(integrator, labels):
    labels = labels.copy()
    np.clip(labels, 0.0, 1.0, out=labels)
    labels /= labels.sum(axis=1)[:, None]
    return integrator.manifold.mean(integrator.quaternions[None,None], labels[None])[0, 0]