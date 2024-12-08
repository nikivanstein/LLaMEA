import numpy as np
import random

class HybridDE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.x = np.random.uniform(-5.0, 5.0, size=(budget, dim))
        self.f_best = np.inf
        self.x_best = None
        self.mut_prob = 0.1
        self.crossover_prob = 0.8
        self.f_eval = 0

    def __call__(self, func):
        for i in range(self.budget):
            # Differential Evolution
            for j in range(self.dim):
                diff = np.zeros(self.dim)
                k = random.randint(0, self.dim - 1)
                l = random.randint(0, self.dim - 1)
                diff[j] = self.x[k, j] - self.x[l, j]
                self.x[i, j] += self.mut_prob * diff * np.sign(np.random.uniform(0, 1))

            # Crossover
            if random.random() < self.crossover_prob:
                j = random.randint(0, self.dim - 1)
                k = random.randint(0, self.dim - 1)
                self.x[i, j], self.x[i, k] = self.x[i, k], self.x[i, j]

            # Evaluate the function
            f_i = func(self.x[i, :])
            self.f_eval += 1
            if f_i < self.f_best:
                self.f_best = f_i
                self.x_best = self.x[i, :]
            if f_i < func(self.x_best, :):
                self.f_best = f_i
                self.x_best = self.x[i, :]

        return self.x_best, self.f_best

# Example usage
def func(x):
    return np.sum(x**2)

hybrid_de = HybridDE(budget=100, dim=10)
x_best, f_best = hybrid_de(func)
print(f_best)
# ```