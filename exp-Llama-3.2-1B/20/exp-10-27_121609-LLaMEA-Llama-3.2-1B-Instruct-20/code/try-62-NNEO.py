import numpy as np
import random
from scipy.optimize import differential_evolution

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

    def mutate(self, x):
        if random.random() < 0.2:
            x[0] = random.uniform(-5.0, 5.0)
        if random.random() < 0.2:
            x[1] = random.uniform(-5.0, 5.0)
        return x

    def evaluate_fitness(self, individual):
        updated_individual = individual
        for i in range(self.dim):
            updated_individual[i] = self.bounds(updated_individual[i])
        fitness = objective(updated_individual)
        self.fitnesses[individual] = fitness
        return fitness

    def bounds(self, individual):
        return (individual.min() - 5.0, individual.max() + 5.0)

    def __str__(self):
        return "NNEO"

# Description: Novel NNEO algorithm that uses differential evolution for optimization.
# Code: 