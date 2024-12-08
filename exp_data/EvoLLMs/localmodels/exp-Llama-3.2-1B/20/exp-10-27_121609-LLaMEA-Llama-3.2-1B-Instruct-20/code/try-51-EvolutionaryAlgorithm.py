import numpy as np
import random

class EvolutionaryAlgorithm:
    def __init__(self, budget, dim, mutation_prob=0.2):
        self.budget = budget
        self.dim = dim
        self.population_size = 50
        self.population = np.random.uniform(-5.0, 5.0, (self.population_size, self.dim))
        self.fitnesses = np.zeros((self.population_size, self.dim))
        self.mutation_prob = mutation_prob

    def __call__(self, func, iterations=1000, max_evals=10000):
        def objective(x):
            return func(x)

        def bounds(x):
            return (x.min() - 5.0, x.max() + 5.0)

        for _ in range(iterations):
            for i in range(self.population_size):
                x = self.population[i]
                fitness = objective(x)
                if fitness < self.fitnesses[i, x] + 1e-6:
                    self.fitnesses[i, x] = fitness
                    self.population[i] = x

            # Select new individuals based on probability
            new_individuals = []
            for _ in range(self.population_size):
                if random.random() < self.mutation_prob:
                    # Randomly select a new individual
                    new_individual = np.random.uniform(-5.0, 5.0, self.dim)
                    new_individuals.append(new_individual)

            # Replace old individuals with new ones
            self.population = np.concatenate((self.population, new_individuals))

            # Evaluate the new population
            new_fitnesses = np.zeros((self.population_size, self.dim))
            for i in range(self.population_size):
                x = self.population[i]
                fitness = objective(x)
                new_fitnesses[i] = fitness

            # Select the best individuals for the next iteration
            new_population = []
            for i in range(self.population_size):
                if new_fitnesses[i] < self.fitnesses[i, x] + 1e-6:
                    new_population.append(x)
                else:
                    new_population.append(new_individuals[i])

            # Replace the old population with the new one
            self.population = new_population

        return self.fitnesses

# Description: Evolutionary Algorithm for BBOB Optimization
# Code: 