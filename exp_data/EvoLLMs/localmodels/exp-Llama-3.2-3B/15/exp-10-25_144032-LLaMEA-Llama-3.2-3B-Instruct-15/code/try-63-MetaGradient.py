import numpy as np
import random

class MetaGradient:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 50
        self.population = np.random.uniform(-5.0, 5.0, (self.population_size, self.dim))
        self.fitness = np.zeros(self.population_size)
        self.best_individual = np.zeros(self.dim)
        self.best_fitness = -np.inf
        self.mutation_prob = 0.15

    def __call__(self, func):
        for _ in range(self.budget):
            # Evaluate the fitness of the current population
            self.fitness = func(self.population)

            # Select the fittest individuals
            parents = np.argsort(self.fitness)[:-int(self.population_size/2):-1]
            self.population = self.population[parents]

            # Perform gradient information
            gradient = np.zeros((self.dim,))
            for i in range(self.population_size):
                gradient += (func(self.population[i]) - func(self.population[parents[np.random.randint(0, self.population_size-1)]])) * (self.population[i] - self.population[parents[np.random.randint(0, self.population_size-1)]]) / (self.population_size - 1)

            # Perform probabilistic mutation
            for i in range(self.population_size):
                if random.random() < self.mutation_prob:
                    mutation = np.random.uniform(-0.1, 0.1, self.dim)
                    self.population[i] += mutation
                    self.population[i] = np.clip(self.population[i], -5.0, 5.0)

            # Calculate the fitness of the mutated population
            self.fitness = func(self.population)

            # Update the population and the best individual
            self.population = np.sort(self.population, axis=0)
            self.best_individual = self.population[0]
            self.best_fitness = func(self.best_individual)

            # Check for convergence
            if self.best_fitness > self.best_fitness:
                break

        return self.best_individual, self.best_fitness

# Example usage:
def func(x):
    return np.sum(x**2)

budget = 100
dim = 10
optimizer = MetaGradient(budget, dim)
best_individual, best_fitness = optimizer(func)
print("Best individual:", best_individual)
print("Best fitness:", best_fitness)