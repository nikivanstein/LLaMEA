import numpy as np
import random

class DEASPS:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 100
        self.adaptive_step_size = 0.5
        self.adaptive_population_size = 0.5
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.mutation_prob = 0.1

    def __call__(self, func):
        population = np.random.uniform(self.lower_bound, self.upper_bound, (self.population_size, self.dim))

        for i in range(self.budget):
            fitness = np.array([func(x) for x in population])
            best_individual = population[np.argmin(fitness)]

            new_population = population + self.adaptive_step_size * np.random.uniform(-1, 1, (self.population_size, self.dim))

            if np.random.rand() < self.adaptive_population_size:
                new_population = new_population[:int(self.population_size * self.adaptive_population_size)]

            # Adaptive mutation
            mutated_population = []
            for individual in population:
                if np.random.rand() < self.mutation_prob:
                    mutated_individual = individual + np.random.uniform(-1, 1, self.dim)
                    mutated_population.append(mutated_individual)
                else:
                    mutated_population.append(individual)

            # Replace worst individual with best individual
            population[np.argsort(fitness)] = np.concatenate((population[np.argsort(fitness)], mutated_population[np.argsort(fitness)]))

        best_x = population[np.argmin(fitness)]
        return best_x

# Example usage:
def func(x):
    return np.sum(x**2)

de = DEASPS(budget=100, dim=10)
best_x = de(func)
print("Best x:", best_x)
print("Best f(x):", func(best_x))