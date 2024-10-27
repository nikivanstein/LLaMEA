import numpy as np
import random
from scipy.optimize import minimize

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

        def objective_new(x):
            return objective(x) - objective_new(x)

        def bounds_new(x):
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
        new_individual = individual.copy()
        for _ in range(self.population_size):
            x = new_individual
            if random.random() < 0.2:
                new_individual = bounds(new_individual)
            if random.random() < 0.2:
                new_individual = objective_new(new_individual)
            if random.random() < 0.2:
                new_individual = bounds_new(new_individual)
        return new_individual

    def evaluate_fitness(self, individual):
        fitness = objective(individual)
        if fitness < self.fitnesses[individual] + 1e-6:
            self.fitnesses[individual] = fitness
        return fitness