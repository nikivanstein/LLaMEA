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

        for _ in range(self.budget):
            for i in range(self.population_size):
                x = self.population[i]
                fitness = objective(x)
                if fitness < self.fitnesses[i, x] + 1e-6:
                    self.fitnesses[i, x] = fitness
                    self.population[i] = x

        return self.fitnesses

    def mutate(self, individual):
        for i in range(self.dim):
            if random.random() < 0.2:
                new_individual = individual.copy()
                new_individual[i] += random.uniform(-0.1, 0.1)
                if new_individual[i] < -5.0:
                    new_individual[i] = -5.0
                elif new_individual[i] > 5.0:
                    new_individual[i] = 5.0
                new_individual[i] = np.clip(new_individual[i], -5.0, 5.0)
                if new_individual not in self.population:
                    self.population.append(new_individual)
        return self.population

    def crossover(self, parent1, parent2):
        child = parent1.copy()
        for i in range(self.dim):
            if random.random() < 0.5:
                child[i] = parent2[i]
        return child

    def evolve(self, func):
        for _ in range(self.budget):
            for i in range(self.population_size):
                x = self.population[i]
                fitness = objective(x)
                if fitness < self.fitnesses[i, x] + 1e-6:
                    self.fitnesses[i, x] = fitness
                    self.population[i] = x
            new_individual = self.mutate(self.population[i])
            self.population[i] = new_individual
            if self.population[i] in self.population:
                self.population.remove(self.population[i])
        return self.fitnesses

# One-line description: Neural Network Evolutionary Optimization algorithm
# Code: 