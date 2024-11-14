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

        def dynamic_update(a, x, alpha, beta, delta):
            A1 = 2 * a * np.random.rand(self.dim) - a
            C1 = 2 * np.random.rand(self.dim)
            D_alpha = np.abs(C1 * alpha - x)
            X1 = alpha - A1 * D_alpha

            A2 = 2 * a * np.random.rand(self.dim) - a
            C2 = 2 * np.random.rand(self.dim)
            D_beta = np.abs(C2 * beta - x)
            X2 = beta - A2 * D_beta

            A3 = 2 * a * np.random.rand(self.dim) - a
            C3 = 2 * np.random.rand(self.dim)
            D_delta = np.abs(C3 * delta - x)
            X3 = delta - A3 * D_delta

            return (X1 + X2 + X3) / 3

        def optimize():
            population = initialize_population()
            alpha, beta, delta = population[np.argsort([func(ind) for ind in population])[:3]]

            for _ in range(self.budget):
                a = 2 - 2 * (_ / self.budget)
                for i in range(self.budget):
                    x = population[i]
                    population[i] = dynamic_update(a, x, alpha, beta, delta)

                alpha, beta, delta = population[np.argsort([func(ind) for ind in population])[:3]]

            return alpha

        return optimize()