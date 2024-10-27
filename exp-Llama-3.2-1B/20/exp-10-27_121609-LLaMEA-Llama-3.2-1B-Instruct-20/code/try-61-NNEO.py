import numpy as np
import random

class NNEO:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 50
        self.population = np.random.uniform(-5.0, 5.0, (self.population_size, self.dim))
        self.fitnesses = np.zeros((self.population_size, self.dim))

    def __call__(self, func):
        def objective(x):
            return func(x)

        def bounds(x):
            return (x.min() - 5.0, x.max() + 5.0)

        def mutate(individual):
            new_individual = individual.copy()
            if random.random() < 0.2:
                new_individual[random.randint(0, self.dim-1)] = random.uniform(-5.0, 5.0)
            return new_individual

        def crossover(parent1, parent2):
            child = parent1.copy()
            if random.random() < 0.2:
                child[random.randint(0, self.dim-1)] = parent2[random.randint(0, self.dim-1)]
            return child

        for _ in range(self.budget):
            for i in range(self.population_size):
                x = self.population[i]
                fitness = objective(x)
                if fitness < self.fitnesses[i, x] + 1e-6:
                    self.fitnesses[i, x] = fitness
                    self.population[i] = x

        # Select the fittest individual
        self.population = self.population[np.argsort(self.fitnesses, axis=1, descending=True)]

        # Evolve the population using mutation and crossover
        while len(self.population) < self.population_size:
            individual = self.population[np.random.randint(0, self.population_size)]
            mutated_individual = mutate(individual)
            if mutated_individual not in self.population:
                self.population.append(mutated_individual)

        # Evaluate the fittest individual
        self.population = self.population[np.argsort(self.fitnesses, axis=1, descending=True)]

        return self.fitnesses

# Description: Novel NNEO Algorithm for Black Box Optimization using Evolutionary Strategies
# Code: