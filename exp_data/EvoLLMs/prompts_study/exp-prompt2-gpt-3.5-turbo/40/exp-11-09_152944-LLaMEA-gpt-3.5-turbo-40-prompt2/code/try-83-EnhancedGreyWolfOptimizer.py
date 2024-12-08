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
                mutated_population = np.zeros((self.budget, self.dim))

                for i in range(self.budget):
                    x = population[i]
                    X1 = alpha - a * np.abs(2 * np.random.rand(self.dim) * alpha - x)
                    X2 = beta - a * np.abs(2 * np.random.rand(self.dim) * beta - x)
                    X3 = delta - a * np.abs(2 * np.random.rand(self.dim) * delta - x)
                    mutated_population[i] = (X1 + X2 + X3) / 3

                combined_population = np.vstack((population, mutated_population))
                fitness_values = np.array([func(ind) for ind in combined_population])
                top_indices = np.argsort(fitness_values)[:self.budget]

                population = combined_population[top_indices]
                alpha, beta, delta = population[:3]

            return alpha

        return optimize()