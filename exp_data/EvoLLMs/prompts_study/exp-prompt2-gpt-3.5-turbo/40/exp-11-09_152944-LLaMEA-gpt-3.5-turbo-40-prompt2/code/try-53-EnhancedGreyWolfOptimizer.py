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
            search_space_factor = 1.0

            for _ in range(self.budget):
                a = 2 - 2 * (_ / self.budget)
                for i in range(self.budget):
                    x = population[i]
                    X1 = alpha - a * np.abs(2 * np.random.rand(self.dim) * alpha - x) * search_space_factor
                    X2 = beta - a * np.abs(2 * np.random.rand(self.dim) * beta - x) * search_space_factor
                    X3 = delta - a * np.abs(2 * np.random.rand(self.dim) * delta - x) * search_space_factor
                    population[i] = (X1 + X2 + X3) / 3

                performance = [func(ind) for ind in population]
                best_idx = np.argmin(performance)
                worst_idx = np.argmax(performance)
                search_space_factor = np.clip(search_space_factor * (1 + 0.01 * (performance[best_idx] - performance[worst_idx])), 0.5, 2.0)
                alpha, beta, delta = population[np.argsort(performance)[:3]]

            return alpha

        return optimize()