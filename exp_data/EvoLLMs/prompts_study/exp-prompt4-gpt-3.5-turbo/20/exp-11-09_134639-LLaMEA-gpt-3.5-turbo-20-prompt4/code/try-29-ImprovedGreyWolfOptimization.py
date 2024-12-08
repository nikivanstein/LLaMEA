import numpy as np

class ImprovedGreyWolfOptimization:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lb = -5.0
        self.ub = 5.0

    def __call__(self, func):
        def initialize_population():
            return np.random.uniform(self.lb, self.ub, (self.budget, self.dim))

        def adaptive_levy_flight():
            beta = 1.5
            sigma = (np.math.gamma(1 + beta) * np.math.sin(np.pi * beta / 2) / np.math.gamma((1 + beta) / 2) / beta / 2 ** ((beta - 1) / 2)) ** (1 / beta
            )
            u = np.random.normal(0, sigma, self.dim)
            v = np.random.normal(0, 1, self.dim)
            step = u / np.abs(v) ** (1 / beta)
            return 0.01 * step * np.sqrt(self.budget / 1000)

        population = initialize_population()
        fitness = [func(individual) for individual in population]
        best_index = np.argmin(fitness)
        best_solution = population[best_index]

        for _ in range(self.budget - self.budget):
            a = 2 - 2 * _ / self.budget
            for i in range(self.budget):
                A = 2 * a * np.random.rand(self.dim) - a
                C = 2 * np.random.rand(self.dim)
                P = np.random.rand(self.dim)

                if i < self.budget / 2:
                    D = np.abs(C * best_solution - population[i])
                    X1 = best_solution - A * D
                    population[i] = np.clip(X1, self.lb, self.ub)
                else:
                    D1 = np.abs(C * best_solution - population[i])
                    X1 = best_solution - A * D1
                    population[i] = np.clip(X1 + adaptive_levy_flight(), self.lb, self.ub)

            fitness = [func(individual) for individual in population]
            new_best_index = np.argmin(fitness)
            if fitness[new_best_index] < fitness[best_index]:
                best_index = new_best_index
                best_solution = population[best_index]

        return best_solution