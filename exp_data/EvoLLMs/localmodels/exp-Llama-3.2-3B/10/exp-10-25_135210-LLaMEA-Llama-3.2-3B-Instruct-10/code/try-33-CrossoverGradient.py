import numpy as np
import scipy.optimize as optimize

class CrossoverGradient:
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
            # Select two random parents
            x1 = np.random.uniform(self.bounds[0][0], self.bounds[0][1], self.dim)
            x2 = np.random.uniform(self.bounds[0][0], self.bounds[0][1], self.dim)

            # Compute gradient
            x_grad1 = np.zeros(self.dim)
            for i in range(self.dim):
                x_plus_epsilon = x1.copy()
                x_plus_epsilon[i] += 1e-6
                f_plus_epsilon = func(x_plus_epsilon)
                x_minus_epsilon = x1.copy()
                x_minus_epsilon[i] -= 1e-6
                f_minus_epsilon = func(x_minus_epsilon)
                x_grad1[i] = (f_plus_epsilon - f_minus_epsilon) / (2 * 1e-6)

            x_grad2 = np.zeros(self.dim)
            for i in range(self.dim):
                x_plus_epsilon = x2.copy()
                x_plus_epsilon[i] += 1e-6
                f_plus_epsilon = func(x_plus_epsilon)
                x_minus_epsilon = x2.copy()
                x_minus_epsilon[i] -= 1e-6
                f_minus_epsilon = func(x_minus_epsilon)
                x_grad2[i] = (f_plus_epsilon - f_minus_epsilon) / (2 * 1e-6)

            # Compute Hessian
            x_hessian1 = np.zeros((self.dim, self.dim, self.dim, self.dim))
            for i in range(self.dim):
                for j in range(self.dim):
                    for k in range(self.dim):
                        x_plus_epsilon = x1.copy()
                        x_plus_epsilon[i] += 1e-6
                        x_plus_epsilon[j] += 1e-6
                        x_plus_epsilon[k] += 1e-6
                        f_plus_epsilon = func(x_plus_epsilon)
                        x_hessian1[i, j, k, i] = (f_plus_epsilon - func(x1)) / (6 * 1e-6**3)
                        x_plus_epsilon = x1.copy()
                        x_plus_epsilon[i] -= 1e-6
                        x_plus_epsilon[j] += 1e-6
                        x_plus_epsilon[k] += 1e-6
                        f_plus_epsilon = func(x_plus_epsilon)
                        x_hessian1[i, j, k, i] -= (f_plus_epsilon - func(x1)) / (6 * 1e-6**3)
                        x_plus_epsilon = x1.copy()
                        x_plus_epsilon[i] += 1e-6
                        x_plus_epsilon[j] -= 1e-6
                        x_plus_epsilon[k] += 1e-6
                        f_plus_epsilon = func(x_plus_epsilon)
                        x_hessian1[i, j, k, i] += (f_plus_epsilon - func(x1)) / (6 * 1e-6**3)
                        x_plus_epsilon = x1.copy()
                        x_plus_epsilon[i] += 1e-6
                        x_plus_epsilon[j] += 1e-6
                        x_plus_epsilon[k] -= 1e-6
                        f_plus_epsilon = func(x_plus_epsilon)
                        x_hessian1[i, j, k, i] -= (f_plus_epsilon - func(x1)) / (6 * 1e-6**3)

            x_hessian2 = np.zeros((self.dim, self.dim, self.dim, self.dim))
            for i in range(self.dim):
                for j in range(self.dim):
                    for k in range(self.dim):
                        x_plus_epsilon = x2.copy()
                        x_plus_epsilon[i] += 1e-6
                        x_plus_epsilon[j] += 1e-6
                        x_plus_epsilon[k] += 1e-6
                        f_plus_epsilon = func(x_plus_epsilon)
                        x_hessian2[i, j, k, i] = (f_plus_epsilon - func(x2)) / (6 * 1e-6**3)
                        x_plus_epsilon = x2.copy()
                        x_plus_epsilon[i] -= 1e-6
                        x_plus_epsilon[j] += 1e-6
                        x_plus_epsilon[k] += 1e-6
                        f_plus_epsilon = func(x_plus_epsilon)
                        x_hessian2[i, j, k, i] -= (f_plus_epsilon - func(x2)) / (6 * 1e-6**3)
                        x_plus_epsilon = x2.copy()
                        x_plus_epsilon[i] += 1e-6
                        x_plus_epsilon[j] -= 1e-6
                        x_plus_epsilon[k] += 1e-6
                        f_plus_epsilon = func(x_plus_epsilon)
                        x_hessian2[i, j, k, i] += (f_plus_epsilon - func(x2)) / (6 * 1e-6**3)
                        x_plus_epsilon = x2.copy()
                        x_plus_epsilon[i] += 1e-6
                        x_plus_epsilon[j] += 1e-6
                        x_plus_epsilon[k] -= 1e-6
                        f_plus_epsilon = func(x_plus_epsilon)
                        x_hessian2[i, j, k, i] -= (f_plus_epsilon - func(x2)) / (6 * 1e-6**3)

            # Compute crossover probability
            p_crossover = np.random.uniform(0, 1)
            if p_crossover < 0.1:
                # Perform crossover
                x_best_new = (x1 + x2) / 2
            else:
                # Perform selection
                x_best_new = x1

            # Compute gradient
            x_grad_best_new = np.zeros(self.dim)
            for i in range(self.dim):
                x_plus_epsilon = x_best_new.copy()
                x_plus_epsilon[i] += 1e-6
                f_plus_epsilon = func(x_plus_epsilon)
                x_minus_epsilon = x_best_new.copy()
                x_minus_epsilon[i] -= 1e-6
                f_minus_epsilon = func(x_minus_epsilon)
                x_grad_best_new[i] = (f_plus_epsilon - f_minus_epsilon) / (2 * 1e-6)

            # Compute Hessian
            x_hessian_best_new = np.zeros((self.dim, self.dim, self.dim, self.dim))
            for i in range(self.dim):
                for j in range(self.dim):
                    for k in range(self.dim):
                        x_plus_epsilon = x_best_new.copy()
                        x_plus_epsilon[i] += 1e-6
                        x_plus_epsilon[j] += 1e-6
                        x_plus_epsilon[k] += 1e-6
                        f_plus_epsilon = func(x_plus_epsilon)
                        x_hessian_best_new[i, j, k, i] = (f_plus_epsilon - func(x_best_new)) / (6 * 1e-6**3)
                        x_plus_epsilon = x_best_new.copy()
                        x_plus_epsilon[i] -= 1e-6
                        x_plus_epsilon[j] += 1e-6
                        x_plus_epsilon[k] += 1e-6
                        f_plus_epsilon = func(x_plus_epsilon)
                        x_hessian_best_new[i, j, k, i] -= (f_plus_epsilon - func(x_best_new)) / (6 * 1e-6**3)
                        x_plus_epsilon = x_best_new.copy()
                        x_plus_epsilon[i] += 1e-6
                        x_plus_epsilon[j] -= 1e-6
                        x_plus_epsilon[k] += 1e-6
                        f_plus_epsilon = func(x_plus_epsilon)
                        x_hessian_best_new[i, j, k, i] += (f_plus_epsilon - func(x_best_new)) / (6 * 1e-6**3)
                        x_plus_epsilon = x_best_new.copy()
                        x_plus_epsilon[i] += 1e-6
                        x_plus_epsilon[j] += 1e-6
                        x_plus_epsilon[k] -= 1e-6
                        f_plus_epsilon = func(x_plus_epsilon)
                        x_hessian_best_new[i, j, k, i] -= (f_plus_epsilon - func(x_best_new)) / (6 * 1e-6**3)

            # Update x_best and x_grad
            x_best_new = x_best_new
            f_best_new = func(x_best_new)
            if f_best_new < self.f_best:
                self.x_best = x_best_new
                self.f_best = f_best_new

            # Update x_grad and x_hessian
            self.x_grad = self.x_grad - np.dot(self.x_hessian_best_new[:, :, :, i], self.x_hessian_best_new[:, :, i, :]) * self.x_hessian_best_new[:, :, i, i]
            self.x_hessian[:, :, i, :] = self.x_hessian[:, :, i, :] / np.linalg.norm(self.x_hessian[:, :, i, :])

# Test the algorithm
def func(x):
    return np.sum(x**2)

budget = 100
dim = 10
alg = CrossoverGradient(budget, dim)
alg(func)