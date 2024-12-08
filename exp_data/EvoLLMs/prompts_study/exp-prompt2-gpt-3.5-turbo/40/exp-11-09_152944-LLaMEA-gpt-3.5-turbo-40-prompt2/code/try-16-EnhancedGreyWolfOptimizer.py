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
                group_size = max(2, int(self.budget * np.sqrt(_ + 1) / self.budget))
                
                for g in range(0, self.budget, group_size):
                    group = population[g:g+group_size]
                    group_alpha, group_beta, group_delta = group[np.argsort([func(ind) for ind in group])[:3]]
                    for i in range(g, g+group_size):
                        x = population[i]
                        X1 = group_alpha - a * np.abs(2 * np.random.rand(self.dim) * group_alpha - x)
                        X2 = group_beta - a * np.abs(2 * np.random.rand(self.dim) * group_beta - x)
                        X3 = group_delta - a * np.abs(2 * np.random.rand(self.dim) * group_delta - x)
                        population[i] = (X1 + X2 + X3) / 3

                alpha, beta, delta = population[np.argsort([func(ind) for ind in population])[:3]]

            return alpha

        return optimize()