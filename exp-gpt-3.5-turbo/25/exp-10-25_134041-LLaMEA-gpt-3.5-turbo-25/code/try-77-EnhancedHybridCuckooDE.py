import numpy as np

class EnhancedHybridCuckooDE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.pa = 0.25
        self.population = np.random.uniform(-5.0, 5.0, (budget, dim))

    def __call__(self, func):
        for _ in range(self.budget):
            candidate = np.random.uniform(-5.0, 5.0, self.dim)
            idx = np.random.randint(self.budget)
            if func(candidate) < func(self.population[idx]):
                self.population[idx] = candidate
                self.pa = np.clip(self.pa * 1.05, 0, 1)
            else:
                if np.random.rand() < self.pa:
                    self.population[idx] = candidate
        return self.population