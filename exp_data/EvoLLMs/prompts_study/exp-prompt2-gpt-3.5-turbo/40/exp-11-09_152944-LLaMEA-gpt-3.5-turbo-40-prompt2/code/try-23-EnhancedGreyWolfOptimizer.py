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
            inertia_weight = 0.5

            for _ in range(self.budget):
                a = 2 - 2 * (_ / self.budget)
                inertia_weight = 0.5 + 0.3 * (_ / self.budget)
                for i in range(self.budget):
                    x = population[i]
                    X1 = alpha - a * np.abs(inertia_weight * np.random.rand(self.dim) * (2 * alpha - x))
                    X2 = beta - a * np.abs(inertia_weight * np.random.rand(self.dim) * (2 * beta - x))
                    X3 = delta - a * np.abs(inertia_weight * np.random.rand(self.dim) * (2 * delta - x))
                    population[i] = (X1 + X2 + X3) / 3

                alpha, beta, delta = population[np.argsort([func(ind) for ind in population])[:3]]

            return alpha

        return optimize()