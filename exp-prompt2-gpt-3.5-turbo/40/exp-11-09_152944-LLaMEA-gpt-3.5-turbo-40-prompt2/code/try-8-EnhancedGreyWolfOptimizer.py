import numpy as np

class EnhancedGreyWolfOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lb = -5.0
        self.ub = 5.0
        self.population_size = 10  # Initial population size

    def __call__(self, func):
        def initialize_population():
            return np.random.uniform(self.lb, self.ub, (self.population_size, self.dim))

        def optimize():
            population = initialize_population()
            alpha, beta, delta = population[np.argsort([func(ind) for ind in population])[:3]]

            for _ in range(self.budget):
                a = 2 - 2 * (_ / self.budget)
                self.population_size = max(3, int(10 * (1 - _ / self.budget)))  # Update population size dynamically
                new_population = np.zeros((self.population_size, self.dim))

                for i in range(self.population_size):
                    x = population[i % self.population_size]
                    X1 = alpha - a * np.abs(2 * np.random.rand(self.dim) * alpha - x)
                    X2 = beta - a * np.abs(2 * np.random.rand(self.dim) * beta - x)
                    X3 = delta - a * np.abs(2 * np.random.rand(self.dim) * delta - x)
                    new_population[i] = (X1 + X2 + X3) / 3

                population[:self.population_size] = new_population
                alpha, beta, delta = population[np.argsort([func(ind) for ind in population])[:3]]

            return alpha

        return optimize()