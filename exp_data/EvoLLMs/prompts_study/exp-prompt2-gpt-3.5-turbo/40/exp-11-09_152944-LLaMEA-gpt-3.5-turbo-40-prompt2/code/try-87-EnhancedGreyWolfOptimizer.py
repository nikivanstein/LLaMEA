import numpy as np

class EnhancedGreyWolfOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lb = -5.0
        self.ub = 5.0

    def __call__(self, func):
        def initialize_population():
            return np.random.uniform(self.lb, self.ub, (self.budget, self.dim))

        def optimize():
            population = initialize_population()
            alpha, beta, delta = population[np.argsort([func(ind) for ind in population])[:3]]

            for _ in range(self.budget):
                a = 2 - 2 * (_ / self.budget)
                for i in range(self.budget):
                    x = population[i]
                    diversity_factor = np.std([func(ind) for ind in population])
                    X1 = alpha - a * np.abs(2 * np.random.rand(self.dim) * alpha - x) * diversity_factor
                    X2 = beta - a * np.abs(2 * np.random.rand(self.dim) * beta - x) * diversity_factor
                    X3 = delta - a * np.abs(2 * np.random.rand(self.dim) * delta - x) * diversity_factor
                    population[i] = (X1 + X2 + X3) / 3

                alpha, beta, delta = population[np.argsort([func(ind) for ind in population])[:3]]

            return alpha

        return optimize()