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
            step_size = 0.5

            for _ in range(self.budget):
                a = 2 - 2 * (_ / self.budget)
                for i in range(self.budget):
                    x = population[i]
                    X1 = alpha - step_size * np.abs(2 * np.random.rand(self.dim) * alpha - x)
                    X2 = beta - step_size * np.abs(2 * np.random.rand(self.dim) * beta - x)
                    X3 = delta - step_size * np.abs(2 * np.random.rand(self.dim) * delta - x)
                    population[i] = (X1 + X2 + X3) / 3

                pop_fitness = [func(ind) for ind in population]
                best_idx = np.argmin(pop_fitness)
                step_size = 0.1 + 0.4 * (1 - pop_fitness[best_idx] / np.max(pop_fitness))
                
                alpha, beta, delta = population[np.argsort(pop_fitness)[:3]]

            return alpha

        return optimize()