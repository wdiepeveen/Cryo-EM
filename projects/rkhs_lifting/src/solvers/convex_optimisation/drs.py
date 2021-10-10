import numpy as np


class Douglas_Rachford_Splitting:
    def __init__(self, proxf=None, proxg=None, x0=None, xref=None, tol=1e-2, max_iter=100):

        # Set P-PDHG basic functions
        self.proxf = proxf
        self.proxg = proxg

        # Set iteration variables
        self.tol = tol
        self.max_iter = max_iter

        # Set initialization
        self.x = x0

        # Error metrics
        # Default start error
        ref_error = np.linalg.norm(self.proxg(2 * self.proxf(xref) - xref) - self.proxf(xref))
        # Initial error
        error = np.linalg.norm(self.proxg(2 * self.proxf(self.x) - self.x) - self.proxf(self.x))

        self.ref_error0 = np.sqrt(ref_error ** 2)
        self.error0 = np.sqrt(error**2)
        print("start from ref error= {}".format(self.ref_error0))
        print("initial error= {}".format(self.error0))

        self.relerror = 1.
        self.ref_relerror = 1.

        self.relerrors = []
        self.ref_relerrors = []
        self.pd_gap = []

    def do_step(self):
        x = self.x + self.proxg(2 * self.proxf(self.x) - self.x) - self.proxf(self.x)
        error = np.linalg.norm(x - self.x)
        self.x = x
        relerror = error / self.error0
        ref_relerror = error / self.ref_error0

        self.relerror = relerror
        self.relerrors.append(relerror)
        print("relerror = {}".format(relerror))
        self.ref_relerror = ref_relerror
        self.ref_relerrors.append(ref_relerror)
        print("normalized_error = {}".format(ref_relerror))

    def solve(self):
        k = 1
        print("DRS iterate = {}".format(self.x[0, 0:5]))
        while self.ref_relerror > self.tol and k <= self.max_iter:
            print("=============== ITERATION {} ===============".format(k))
            self.do_step()
            k += 1
            print("DRS iterate = {}".format(self.x[0,0:5]))
