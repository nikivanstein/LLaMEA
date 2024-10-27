import numpy as np
import scipy.optimize as optimize

class LeverageGradient:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.bounds = [(-5.0, 5.0)] * dim
        self.x_best = np.random.uniform(-5.0, 5.0, dim)
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

# Refine the Leverage-Gradient algorithm with a 0.1 probability to change individual lines
def refine_leverage_gradient(x, x_grad, x_hessian, bounds):
    if np.random.rand() < 0.1:
        i = np.random.randint(0, len(bounds))
        j = np.random.randint(0, len(bounds[i]))
        k = np.random.randint(0, len(bounds[i][j]))
        x[i, j, k] += np.random.uniform(-1, 1)
        x_grad[i, j, k] += np.random.uniform(-1, 1)
        x_hessian[i, j, k, i] += np.random.uniform(-1, 1)
        x_hessian[i, j, k, j] += np.random.uniform(-1, 1)
        x_hessian[i, j, k, k] += np.random.uniform(-1, 1)

# Test the refined algorithm
def func(x):
    return np.sum(x**2)

budget = 100
dim = 10
alg = LeverageGradient(budget, dim)
refined_alg = LeverageGradient(budget, dim)
for _ in range(10):
    x = np.random.uniform(-5.0, 5.0, (dim, dim, dim))
    x_grad = np.zeros((dim, dim, dim))
    x_hessian = np.zeros((dim, dim, dim, dim))
    refined_x = np.random.uniform(-5.0, 5.0, (dim, dim, dim))
    refined_x_grad = np.zeros((dim, dim, dim))
    refined_x_hessian = np.zeros((dim, dim, dim, dim))
    for _ in range(10):
        x = refined_x
        x_grad = refined_x_grad
        x_hessian = refined_x_hessian
        f = func(x)
        if f < alg.f_best:
            alg.x_best = x
            alg.f_best = f
        if f < refined_alg.f_best:
            refined_alg.x_best = x
            refined_alg.f_best = f
        # Compute gradient
        for i in range(dim):
            for j in range(dim):
                for k in range(dim):
                    x_plus_epsilon = x.copy()
                    x_plus_epsilon[i, j, k] += 1e-6
                    f_plus_epsilon = func(x_plus_epsilon)
                    x_grad[i, j, k] = (f_plus_epsilon - func(x)) / (2 * 1e-6)
                    x_plus_epsilon = x.copy()
                    x_plus_epsilon[i, j, k] -= 1e-6
                    f_plus_epsilon = func(x_plus_epsilon)
                    x_grad[i, j, k] -= (f_plus_epsilon - func(x)) / (2 * 1e-6)
        # Compute Hessian
        for i in range(dim):
            for j in range(dim):
                for k in range(dim):
                    for l in range(dim):
                        x_plus_epsilon = x.copy()
                        x_plus_epsilon[i, j, k] += 1e-6
                        x_plus_epsilon[i, j, l] += 1e-6
                        f_plus_epsilon = func(x_plus_epsilon)
                        x_hessian[i, j, k, l] = (f_plus_epsilon - func(x)) / (6 * 1e-6**3)
                        x_plus_epsilon = x.copy()
                        x_plus_epsilon[i, j, k] -= 1e-6
                        x_plus_epsilon[i, j, l] -= 1e-6
                        f_plus_epsilon = func(x_plus_epsilon)
                        x_hessian[i, j, k, l] -= (f_plus_epsilon - func(x)) / (6 * 1e-6**3)
                        x_plus_epsilon = x.copy()
                        x_plus_epsilon[i, j, k] += 1e-6
                        x_plus_epsilon[i, j, l] += 1e-6
                        f_plus_epsilon = func(x_plus_epsilon)
                        x_hessian[i, j, k, l] += (f_plus_epsilon - func(x)) / (6 * 1e-6**3)
                        x_plus_epsilon = x.copy()
                        x_plus_epsilon[i, j, k] -= 1e-6
                        x_plus_epsilon[i, j, l] += 1e-6
                        f_plus_epsilon = func(x_plus_epsilon)
                        x_hessian[i, j, k, l] -= (f_plus_epsilon - func(x)) / (6 * 1e-6**3)
                        x_plus_epsilon = x.copy()
                        x_plus_epsilon[i, j, k] += 1e-6
                        x_plus_epsilon[i, j, l] -= 1e-6
                        f_plus_epsilon = func(x_plus_epsilon)
                        x_hessian[i, j, k, l] += (f_plus_epsilon - func(x)) / (6 * 1e-6**3)
                        x_plus_epsilon = x.copy()
                        x_plus_epsilon[i, j, k] += 1e-6
                        x_plus_epsilon[i, j, l] += 1e-6
                        f_plus_epsilon = func(x_plus_epsilon)
                        x_hessian[i, j, k, l] -= (f_plus_epsilon - func(x)) / (6 * 1e-6**3)
        # Update x_best and x_grad
        x_best_new = x + np.dot(x_grad, x_hessian[:, :, :, i]) * 1e-6
        f_best_new = func(x_best_new)
        if f_best_new < alg.f_best:
            alg.x_best = x_best_new
            alg.f_best = f_best_new
        if f_best_new < refined_alg.f_best:
            refined_alg.x_best = x_best_new
            refined_alg.f_best = f_best_new
        # Update x_grad and x_hessian
        x_grad = x_grad - np.dot(x_hessian[:, :, :, i], x_hessian[:, :, :, i]) * x_hessian[:, :, :, i]
        x_hessian[:, :, :, i] = x_hessian[:, :, :, i] / np.linalg.norm(x_hessian[:, :, :, i])
        refined_x = refined_x + np.dot(refined_x_grad, refined_x_hessian[:, :, :, i]) * 1e-6
        refined_x_grad = refined_x_grad - np.dot(refined_x_hessian[:, :, :, i], refined_x_hessian[:, :, :, i]) * refined_x_hessian[:, :, :, i]
        refined_x_hessian[:, :, :, i] = refined_x_hessian[:, :, :, i] / np.linalg.norm(refined_x_hessian[:, :, :, i])
    # Refine the Leverage-Gradient algorithm with a 0.1 probability to change individual lines
    refined_x = refined_x
    refined_x_grad = refined_x_grad
    refined_x_hessian = refined_x_hessian
    for i in range(10):
        refined_x[i, j, k] += np.random.uniform(-1, 1)
        refined_x_grad[i, j, k] += np.random.uniform(-1, 1)
        refined_x_hessian[i, j, k, i] += np.random.uniform(-1, 1)
        refined_x_hessian[i, j, k, j] += np.random.uniform(-1, 1)
        refined_x_hessian[i, j, k, k] += np.random.uniform(-1, 1)