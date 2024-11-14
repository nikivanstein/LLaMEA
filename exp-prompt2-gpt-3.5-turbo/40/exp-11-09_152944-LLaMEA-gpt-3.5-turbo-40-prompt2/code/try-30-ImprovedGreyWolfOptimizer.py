import numpy as np

class ImprovedGreyWolfOptimizer:
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
                    X1 = alpha - a * np.abs(2 * np.random.rand(self.dim) * alpha - x)
                    X2 = beta - a * np.abs(2 * np.random.rand(self.dim) * beta - x)
                    X3 = delta - a * np.abs(2 * np.random.rand(self.dim) * delta - x)
                    population[i] = (X1 + X2 + X3) / 3

                    # Opposition-based learning
                    opposite_x = self.lb + self.ub - x
                    opposite_X1 = alpha - a * np.abs(2 * np.random.rand(self.dim) * alpha - opposite_x)
                    opposite_X2 = beta - a * np.abs(2 * np.random.rand(self.dim) * beta - opposite_x)
                    opposite_X3 = delta - a * np.abs(2 * np.random.rand(self.dim) * delta - opposite_x)
                    opposite_X = (opposite_X1 + opposite_X2 + opposite_X3) / 3

                    population[i] = 0.5 * (population[i] + opposite_X)

                alpha, beta, delta = population[np.argsort([func(ind) for ind in population])[:3]]

            return alpha

        return optimize()