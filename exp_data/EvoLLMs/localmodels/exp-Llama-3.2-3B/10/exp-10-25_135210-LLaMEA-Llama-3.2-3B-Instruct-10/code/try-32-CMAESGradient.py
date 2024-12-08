import numpy as np
from scipy.optimize import minimize

class CMAESGradient:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.bounds = [(-5.0, 5.0)] * dim
        self.x_best = np.random.uniform(self.bounds[0][0], self.bounds[0][1], dim)
        self.f_best = np.inf
        self.cma_x = np.zeros((self.dim, self.dim))
        self.cma_mu = np.zeros(self.dim)
        self.cma_sigma = np.zeros(self.dim)

    def __call__(self, func):
        for _ in range(self.budget):
            res = minimize(lambda x: func(x), self.x_best, method="SLSQP", bounds=self.bounds)
            if res.fun < self.f_best:
                self.x_best = res.x
                self.f_best = res.fun

            # Compute gradient
            gradient = np.zeros(self.dim)
            for i in range(self.dim):
                gradient[i] = (func(self.x_best + 1e-6 * np.eye(self.dim)[i, :]) - func(self.x_best - 1e-6 * np.eye(self.dim)[i, :])) / (2 * 1e-6)

            # Update cma_x, cma_mu, and cma_sigma
            self.cma_x = self.cma_x - self.cma_mu * self.cma_sigma
            self.cma_mu = self.cma_mu - np.dot(self.cma_x.T, self.cma_x) / self.dim
            self.cma_sigma = self.cma_sigma + 1 / self.dim * np.eye(self.dim)

            # Sample new x using cma_x and cma_mu
            x_new = self.cma_x + np.random.normal(0, self.cma_sigma) * np.sqrt(1 + 2 / self.dim)
            f_new = func(x_new)

            # Update x_best and f_best
            if f_new < self.f_best:
                self.x_best = x_new
                self.f_best = f_new

# Test the algorithm
def func(x):
    return np.sum(x**2)

budget = 100
dim = 10
alg = CMAESGradient(budget, dim)
alg()