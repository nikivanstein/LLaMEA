import numpy as np
import scipy.optimize as optimize
import random

class LeverageGradient:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.bounds = [(-5.0, 5.0)] * dim
        self.x_best = np.random.uniform(self.bounds[0][0], self.bounds[0][1], dim)
        self.f_best = np.inf
        self.x_grad = np.zeros((dim, dim))
        self.x_hessian = np.zeros((dim, dim, dim, dim))

    def __call__(self, func):
        for _ in range(self.budget):
            x = np.random.uniform(self.bounds[0][0], self.bounds[0][1], self.dim)
            f = func(x)

            if f < self.f_best:
                self.x_best = x
                self.f_best = f

            # Compute gradient
            x_grad = np.zeros(self.dim)
            for i in range(self.dim):
                x_plus_epsilon = x.copy()
                x_plus_epsilon[i] += 1e-6
                f_plus_epsilon = func(x_plus_epsilon)
                x_minus_epsilon = x.copy()
                x_minus_epsilon[i] -= 1e-6
                f_minus_epsilon = func(x_minus_epsilon)
                x_grad[i] = (f_plus_epsilon - f_minus_epsilon) / (2 * 1e-6)

            # Compute Hessian
            x_hessian = np.zeros((self.dim, self.dim, self.dim, self.dim))
            for i in range(self.dim):
                for j in range(self.dim):
                    for k in range(self.dim):
                        x_plus_epsilon = x.copy()
                        x_plus_epsilon[i] += 1e-6
                        x_plus_epsilon[j] += 1e-6
                        x_plus_epsilon[k] += 1e-6
                        f_plus_epsilon = func(x_plus_epsilon)
                        x_hessian[i, j, k, i] = (f_plus_epsilon - func(x)) / (6 * 1e-6**3)
                        x_plus_epsilon = x.copy()
                        x_plus_epsilon[i] -= 1e-6
                        x_plus_epsilon[j] += 1e-6
                        x_plus_epsilon[k] += 1e-6
                        f_plus_epsilon = func(x_plus_epsilon)
                        x_hessian[i, j, k, i] -= (f_plus_epsilon - func(x)) / (6 * 1e-6**3)
                        x_plus_epsilon = x.copy()
                        x_plus_epsilon[i] += 1e-6
                        x_plus_epsilon[j] -= 1e-6
                        x_plus_epsilon[k] += 1e-6
                        f_plus_epsilon = func(x_plus_epsilon)
                        x_hessian[i, j, k, i] += (f_plus_epsilon - func(x)) / (6 * 1e-6**3)
                        x_plus_epsilon = x.copy()
                        x_plus_epsilon[i] += 1e-6
                        x_plus_epsilon[j] += 1e-6
                        x_plus_epsilon[k] -= 1e-6
                        f_plus_epsilon = func(x_plus_epsilon)
                        x_hessian[i, j, k, i] -= (f_plus_epsilon - func(x)) / (6 * 1e-6**3)

            # Update x_best and x_grad
            x_best_new = x + np.dot(self.x_grad, self.x_hessian[:, :, i, :]) * 1e-6
            f_best_new = func(x_best_new)
            if f_best_new < self.f_best:
                self.x_best = x_best_new
                self.f_best = f_best_new

            # Update x_grad and x_hessian
            self.x_grad = self.x_grad - np.dot(self.x_hessian[:, :, i, :], self.x_hessian[:, :, i, :]) * self.x_hessian[:, :, i, i]
            self.x_hessian[:, :, i, :] = self.x_hessian[:, :, i, :] / np.linalg.norm(self.x_hessian[:, :, i, :])

# Test the algorithm
def func(x):
    return np.sum(x**2)

budget = 100
dim = 10
alg = LeverageGradient(budget, dim)
for i in range(100):
    if random.random() < 0.1:
        # Randomly change the individual lines of the selected solution
        x = np.random.uniform(self.bounds[0][0], self.bounds[0][1], self.dim)
        f = func(x)
        if f < alg.f_best:
            alg.x_best = x
            alg.f_best = f

    alg()