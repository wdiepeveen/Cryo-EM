import numpy as np


class P_PDHG:
    def __init__(self, primalProx, dualProx, operator, adjoint, sigmas, taus, x0, y0, tol=1e-2, max_iter=100):
        # Set P-PDHG basic functions

        # self.pd_gap = pdGap

        self.proxf = primalProx
        self.proxg = dualProx

        self.operator = operator
        self.adjoint_operator = adjoint

        self.sigmas = sigmas
        self.taus = taus

        # Set iteration variables

        self.tol = tol
        self.max_iter = max_iter

        # Set initialization
        self.x0 = x0
        self.y0 = y0

        self.x = x0
        self.xb = x0

        self.y = y0

        # TODO cost function and or primal dual gap
        # Error metrics
        # Default start error
        xstart0_error = np.linalg.norm(self.proxf(np.zeros(self.x.shape) - self.sigmas * self.adjoint_operator(np.zeros(self.y.shape)), self.sigmas))
        ystart0_error = np.linalg.norm(self.proxg(np.zeros(self.y.shape) + self.taus * self.operator(np.zeros(self.x.shape)), self.taus))
        # Initial error
        x_error = np.linalg.norm(self.x - self.proxf(self.x - self.sigmas * self.adjoint_operator(self.y), self.sigmas))
        y_error = np.linalg.norm(self.y - self.proxg(self.y + self.taus * self.operator(self.x), self.taus))

        self.start0_error = np.sqrt(xstart0_error ** 2 + ystart0_error ** 2)
        self.error0 = np.sqrt(x_error**2 + y_error**2)
        print("start from 0 error= {}".format(self.start0_error))
        print("initial error= {}".format(self.error0))

        self.relerror = 1.
        self.normalized_error = 1.

        self.relerrors = []
        self.normalized_errors = []
        self.pd_gap = []

    def do_step(self):
        # print("sigma = {}".format(self.sigma))
        # print("tau = {}".format(self.tau))

        y = self.proxg(self.y + self.taus * self.operator(self.xb), self.taus)
        y_error = np.linalg.norm(y - self.y)

        # y1 = D * x̄
        # y2 = y1
        # y3 = y + τ * y2
        # y4 = y3
        # y5 = proxg(y4, τ)

        x = self.proxf(self.x - self.sigmas * self.adjoint_operator(y), self.sigmas)
        x_error = np.linalg.norm(x - self.x)

        # z1 = y5
        # z2 = -σ * Dadj * z1
        # z3 = z2
        # x1 = x + z2
        # x2 = proxf(x1, σ)

        self.xb = 2 * x - self.x
        self.x = x
        self.y = y

        error = np.sqrt(x_error**2 + y_error**2)
        relerror = error / self.error0
        normalized_error = error / self.start0_error

        self.relerror = relerror
        self.relerrors.append(relerror)
        print("relerror = {}".format(relerror))
        self.normalized_error = normalized_error
        self.normalized_errors.append(normalized_error)
        print("normalized_error = {}".format(normalized_error))

        # x̄ = x2 + θ * (x2 - x)

    def solve(self):
        k = 1
        print("primal iterate = {}".format(self.x[0, 0:5]))
        print("dual iterate = {}".format(self.y[0, 0:5]))
        while self.normalized_error > self.tol and k <= self.max_iter:
            print("=============== ITERATION {} ===============".format(k))
            self.do_step()
            k += 1
            print("primal iterate = {}".format(self.x[0,0:5]))
            print("dual iterate = {}".format(self.y[0,0:5]))
            print("barred primal iterate = {}".format(self.xb[0, 0:5]))
