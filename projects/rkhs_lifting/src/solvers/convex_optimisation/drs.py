import numpy as np


class Douglas_Rachford_Splitting:
    def __init__(self, proxf=None, proxg=None, x0=None, tol=1e-2, max_iter=200):

        # Set P-PDHG basic functions
        self.proxf = proxf
        self.proxg = proxg

        # Set iteration variables
        self.tol = tol
        self.max_iter = max_iter

        # Set initialization
        self.x = x0

        # TODO cost function and or primal dual gap
        # Error metrics
        # Default start error
        start0_error = np.linalg.norm(self.proxg(2 * self.proxf(np.zeros(self.x.shape))) - self.proxf(np.zeros(self.x.shape)))
        # Initial error
        error = np.linalg.norm(self.proxg(2 * self.proxf(self.x) - self.x) - self.proxf(self.x))

        self.start0_error = np.sqrt(start0_error ** 2)
        self.error0 = np.sqrt(error**2)
        print("start from 0 error= {}".format(self.start0_error))
        print("initial error= {}".format(self.error0))

        self.relerror = 1.
        self.normalized_error = 1.

        self.relerrors = []
        self.normalized_errors = []
        self.pd_gap = []

    def do_step(self):
        x = self.x + self.proxg(2 * self.proxf(self.x) - self.x) - self.proxf(self.x)
        error = np.linalg.norm(x - self.x)
        self.x = x
        relerror = error / self.error0
        normalized_error = error / self.start0_error

        self.relerror = relerror
        self.relerrors.append(relerror)
        print("relerror = {}".format(relerror))
        self.normalized_error = normalized_error
        self.normalized_errors.append(normalized_error)
        print("normalized_error = {}".format(normalized_error))

    def solve(self):
        k = 1
        print("DRS iterate = {}".format(self.x[0, 0:5]))
        while self.normalized_error > self.tol and k <= self.max_iter:
            print("=============== ITERATION {} ===============".format(k))
            self.do_step()
            k += 1
            print("DRS iterate = {}".format(self.x[0,0:5]))
