import numpy as np
from scipy.special import gammaln


def vallee_poussin(omega, kappa=10):
    res = np.cos(omega/2)**(2*kappa)
    normalisation = np.sqrt(np.pi) * np.exp(gammaln(kappa+2) - gammaln(kappa + 1/2))
    return normalisation * res