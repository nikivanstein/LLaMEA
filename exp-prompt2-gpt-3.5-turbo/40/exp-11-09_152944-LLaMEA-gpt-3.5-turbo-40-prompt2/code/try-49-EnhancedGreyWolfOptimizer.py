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

            for _ in range(self.budget):
                alpha, beta, delta = population[np.argsort([func(ind) for ind in population])[:3]]
                a = 2 - 2 * (_ / self.budget)

                group_size = max(2, int(self.budget * 0.1))  # Dynamic grouping
                groups = [population[i:i+group_size] for i in range(0, self.budget, group_size)]

                for group in groups:
                    for i in range(len(group)):
                        x = group[i]
                        X1 = alpha - a * np.abs(2 * np.random.rand(self.dim) * alpha - x)
                        X2 = beta - a * np.abs(2 * np.random.rand(self.dim) * beta - x)
                        X3 = delta - a * np.abs(2 * np.random.rand(self.dim) * delta - x)
                        group[i] = (X1 + X2 + X3) / 3

                population = np.concatenate(groups)

            alpha, _, _ = population[np.argsort([func(ind) for ind in population])[:3]]
            return alpha

        return optimize()