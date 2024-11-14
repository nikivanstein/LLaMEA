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
            a = 2

            for _ in range(self.budget):
                a = 2 - 2 * (_ / self.budget)
                for i in range(len(population)):
                    x = population[i]
                    X1 = alpha - a * np.abs(2 * np.random.rand(self.dim) * alpha - x)
                    X2 = beta - a * np.abs(2 * np.random.rand(self.dim) * beta - x)
                    X3 = delta - a * np.abs(2 * np.random.rand(self.dim) * delta - x)
                    new_wolf = (X1 + X2 + X3) / 3
                    if func(new_wolf) < func(population[i]):
                        population[i] = new_wolf

                if _ % 10 == 0 and len(population) <= 2 * self.budget:  # Dynamic population size adaptation
                    new_wolves = initialize_population()
                    population = np.vstack((population, new_wolves))

                alpha, beta, delta = population[np.argsort([func(ind) for ind in population])[:3]]

            return alpha

        return optimize()