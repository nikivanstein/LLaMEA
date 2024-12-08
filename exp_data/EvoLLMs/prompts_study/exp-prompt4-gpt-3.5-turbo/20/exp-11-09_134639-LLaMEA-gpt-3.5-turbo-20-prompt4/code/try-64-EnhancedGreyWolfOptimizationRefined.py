import numpy as np
import chaospy as cp

class EnhancedGreyWolfOptimizationRefined:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lb = -5.0
        self.ub = 5.0

    def __call__(self, func):
        def initialize_population(size):
            return np.random.uniform(self.lb, self.ub, (size, self.dim))

        def chaotic_map_update(x):
            return 3.9 * x * (1 - x)

        population_size = 5
        population = initialize_population(population_size)
        fitness = [func(individual) for individual in population]
        best_index = np.argmin(fitness)
        best_solution = population[best_index]

        for _ in range(self.budget - population_size):
            a = 2 - 2 * _ / self.budget
            for i in range(population_size):
                A = 2 * a * np.random.rand(self.dim) - a
                C = 2 * np.random.rand(self.dim)
                P = np.random.rand(self.dim)

                if i < population_size / 2:
                    D = np.abs(C * best_solution - population[i])
                    X1 = best_solution - A * D
                    population[i] = np.clip(X1, self.lb, self.ub)
                else:
                    D1 = np.abs(C * best_solution - population[i])
                    X1 = best_solution - A * D1
                    population[i] = np.clip(X1 * chaotic_map_update(P), self.lb, self.ub)

            fitness = [func(individual) for individual in population]
            new_best_index = np.argmin(fitness)
            if fitness[new_best_index] < fitness[best_index]:
                best_index = new_best_index
                best_solution = population[best_index]

            if np.random.rand() < 0.2:
                population_size += 2
                population = np.vstack((population, initialize_population(2)))
                fitness.extend([func(individual) for individual in population[-2:]])

        return best_solution