import numpy as np

from aspire.utils.coor_trans import grid_3d
from scipy.fft import ifftshift

def construct_reg_kernel(L,dtype=np.float32):
    # use grid_3d from aspire.utils.coord_trans - also see aspire.volume.rotated_grids
    # multiply by pi
    # Compute function on these indices
    # TODO checkout how we get (2L)**3 sized matrix from this
    #  - A: don;t worry about this. Is probably just a numerical error thing
    #  -- when fft'en vol we also get a higher resolution (se aspire._.kernel l.103)
    #  - see anufft for this?
    #  - checkout documentation on FFT
    # TODO We only have to compute this once and then save it -> then just add it in the solver

    grid3d = grid_3d(2 * L) #, shifted=True)
    x = np.pi * grid3d["x"]
    y = np.pi * grid3d["y"]
    z = np.pi * grid3d["z"]
    kernel = ifftshift(np.sqrt(x ** 2 + y ** 2 + z ** 2).astype(dtype), axes=(0, 1, 2))
    print(kernel[0,:,:])

    # Construct kernel upper left corner? -> can get mesh and construct kernel normally right away



# Test function
L = 5
construct_reg_kernel(L)

# TODO check whether there is a difference between getting a 2L filter right away or by doing fft/ifft