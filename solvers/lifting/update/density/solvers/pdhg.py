import numpy as np


class PDHG:
    def __init__(self, primalProx, dualProx, operator, adjoint, sigma, tau, gamma, x0, y0, tol=1e-3, max_iter=10):
        # Set PDHG basic functions

        self.proxf = primalProx
        self.proxg = dualProx

        self.operator = operator
        self.adjoint_operator = adjoint

        self.sigma = sigma
        self.tau = tau
        self.gamma = gamma

        # Set iteration variables

        self.tol = tol
        self.max_iter = max_iter

        # Set initialization

        self.x = x0

        self.y = y0

    def do_step(self):
        self.y = self.proxg(self.y + self.tau * self.operator(self.xb), self.tau)

        # y1 = D * x̄
        # y2 = y1
        # y3 = y + τ * y2
        # y4 = y3
        # y5 = proxg(y4, τ)

        x = self.proxf(self.x - self.sigma * self.adjoint_operator(self.y), self.sigma)

        # z1 = y5
        # z2 = -σ * Dadj * z1
        # z3 = z2
        # x1 = x + z2
        # x2 = proxf(x1, σ)

        self.theta = 1 / np.sqrt(1 + 2 * self.gamma * self.tau)
        self.sigma = self.theta * self.sigma
        self.tau = self.tau / self.theta

        # θ = 1 / sqrt(1 + 2 * γ * τ)
        # σ = θ * σ
        # τ = τ / θ

        self.xb = x + self.theta * (x - self.x)
        self.x = x

        # x̄ = x2 + θ * (x2 - x)

    def solve(self):
        k = 1
        relerror = 1
        while relerror > self.tol and k <= self.max_iter:
            self.do_step()
            k += 1
            # TODO write relerror update
