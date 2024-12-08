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
        self.mutation_probability = 0.1

    def __call__(self, func):
        population = np.random.uniform(self.lower_bound, self.upper_bound, (self.population_size, self.dim))
        for i in range(self.budget):
            fitness = np.array([func(x) for x in population])
            best_individual = population[np.argmin(fitness)]
            new_population = population + self.adaptive_step_size * np.random.uniform(-1, 1, (self.population_size, self.dim))
            if np.random.rand() < self.adaptive_population_size:
                new_population = new_population[:int(self.population_size * self.adaptive_population_size)]
            for j in range(self.population_size):
                if np.random.rand() < self.mutation_probability:
                    new_individual = self.mutate(new_population[j])
                    new_population[j] = new_individual
            population[np.argsort(fitness)] = np.concatenate((population[np.argsort(fitness)], new_population[np.argsort(fitness)]))
        return population[np.argmin(fitness)]

    def mutate(self, individual):
        mutated_individual = individual + np.random.uniform(-0.1, 0.1, self.dim)
        return np.clip(mutated_individual, self.lower_bound, self.upper_bound)

# Example usage:
def func(x):
    return np.sum(x**2)

de = DEASPS(budget=100, dim=10)
best_x = de(func)
print("Best x:", best_x)
print("Best f(x):", func(best_x))