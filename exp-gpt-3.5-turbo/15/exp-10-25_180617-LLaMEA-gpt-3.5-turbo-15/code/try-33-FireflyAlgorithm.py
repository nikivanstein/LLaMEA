import numpy as np

class FireflyAlgorithm:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.pop_size = 20
        self.max_iter = budget // self.pop_size

    def __call__(self, func):
        lb = -5.0
        ub = 5.0
        pop = lb + (ub - lb) * np.random.rand(self.pop_size, self.dim)
        alpha = 0.2
        beta0 = 1.0
        gamma = 0.97

        for _ in range(self.max_iter):
            for i in range(self.pop_size):
                for j in range(self.pop_size):
                    if func(pop[j]) < func(pop[i]):
                        beta = beta0 * np.exp(-gamma * np.linalg.norm(pop[j] - pop[i])**2)
                        pop[i] += alpha * (pop[j] - pop[i]) + beta * np.random.uniform(-1, 1, self.dim)
                        pop[i] = np.clip(pop[i], lb, ub)

        return pop[func(pop).argmin()]