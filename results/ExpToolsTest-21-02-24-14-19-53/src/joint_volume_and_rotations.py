import logging
import os

import mrcfile
import numpy as np

from aspire.volume import Volume

from pymanopt.manifolds import Rotations

# Own functions
from tools.exp_tools import exp_open


logger = logging.getLogger(__name__)

DATA_DIR = os.path.join(os.path.dirname(__file__), "results/")
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results/")

logger.info(
    "This script illustrates orientation refinement using "
    "lRCPA as primal-dual splitting method"
)

# Define a precision for this experiment
dtype = np.float32

# Load data
infile = mrcfile.open(os.path.join(DATA_DIR, "reconstructed70SRibosome_vol_65p.mrc"))
vol = Volume(infile.data.astype(dtype)) # Load initial volume
sim = exp_open(os.path.join(DATA_DIR, "sim_up.pkl")) # Load simulator
g = sim.rots # Load initial rotations
num_ims = sim.n

# setup manifold

# Manifold of 3 x 3 orthogonal matrices with
# determinant 1.
manifold = Rotations(3, num_ims)

# TODO setup functions
def cost(u,g):
    return 1.


# cF = p -> ManTV(M,data,α,p,anisotropic)
# pP = (p,σ) -> proxDistance(M,σ/α,data,p)
# dP = (ξ,τ) -> ManTVproxDual(N,n,ξ,anisotropic)
# Λ = p -> TBPoint(ManTVdualbasepoint(p),forwardLogs(M,p))
# DΛ = (p,η) -> TBTVector(zeroTVector(N.manifold,getBase(n)),DforwardLogs(M,p,η))
# DΛadj = (p,η) -> AdjDforwardLogs(M,p,getTangent(η))

# pr = lrcpaProblem(M,m,N,n,cF,pP,dP,Λ,DΛ,DΛadj)

# TODO setup solver

# TODO run solver
# fR,r = lRCPA(pr,data)

