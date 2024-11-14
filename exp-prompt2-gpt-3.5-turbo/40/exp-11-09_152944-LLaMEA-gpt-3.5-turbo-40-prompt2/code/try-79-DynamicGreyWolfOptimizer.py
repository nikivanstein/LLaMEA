import numpy as np

class DynamicGreyWolfOptimizer:
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
            c1, c2 = 2.0, 0.5

            for _ in range(self.budget):
                a = 2 - 2 * (_ / self.budget)
                c1 = max(2.0 - 1.5 * (_ / self.budget), 0.5)
                c2 = min(0.5 + 1.5 * (_ / self.budget), 2.0)

                for i in range(self.budget):
                    x = population[i]
                    X1 = alpha - c1 * np.abs(2 * np.random.rand(self.dim) * alpha - x)
                    X2 = beta - c2 * np.abs(2 * np.random.rand(self.dim) * beta - x)
                    X3 = delta - a * np.abs(2 * np.random.rand(self.dim) * delta - x)
                    population[i] = (X1 + X2 + X3) / 3

                alpha, beta, delta = population[np.argsort([func(ind) for ind in population])[:3]]

            return alpha

        return optimize()