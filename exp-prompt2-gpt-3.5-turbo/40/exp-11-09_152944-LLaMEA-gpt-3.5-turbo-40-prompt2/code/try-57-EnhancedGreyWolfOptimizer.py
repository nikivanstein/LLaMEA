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
                for i in range(len(population)): # Change 40.0% of code: Use dynamic population size
                    x = population[i]
                    selected_wolves = population[np.random.choice(len(population), 3, replace=False)]
                    X1 = alpha - a * np.abs(2 * np.random.rand(self.dim) * selected_wolves[0] - x)
                    X2 = beta - a * np.abs(2 * np.random.rand(self.dim) * selected_wolves[1] - x)
                    X3 = delta - a * np.abs(2 * np.random.rand(self.dim) * selected_wolves[2] - x)
                    population[i] = (X1 + X2 + X3) / 3

                alpha, beta, delta = population[np.argsort([func(ind) for ind in population])[:3]]

            return alpha

        return optimize()