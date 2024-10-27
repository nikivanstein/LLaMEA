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

        for _ in range(self.budget):
            for i in range(self.population_size):
                x = self.population[i]
                fitness = objective(x)
                if fitness < self.fitnesses[i, x] + 1e-6:
                    self.fitnesses[i, x] = fitness
                    self.population[i] = x

        return self.fitnesses

    def mutate(self, individual):
        bounds = bounds(individual)
        if random.random() < 0.2:
            new_dim = random.randint(1, self.dim)
            new_individual = individual[:new_dim] + [random.uniform(-5.0, 5.0)]
            self.population[random.randint(0, self.population_size - 1)] = new_individual
            bounds[new_individual] = (bounds[new_individual].min() - 5.0, bounds[new_individual].max() + 5.0)
        elif random.random() < 0.2:
            new_individual = individual + [random.uniform(-5.0, 5.0)]
            self.population[random.randint(0, self.population_size - 1)] = new_individual
            bounds[new_individual] = (bounds[new_individual].min() - 5.0, bounds[new_individual].max() + 5.0)

    def evolve(self, func):
        while True:
            new_population = self.population.copy()
            for _ in range(self.budget):
                new_individual = new_population[random.randint(0, self.population_size - 1)]
                new_fitness = func(new_individual)
                if new_fitness < self.fitnesses[new_individual, new_individual] + 1e-6:
                    self.fitnesses[new_individual, new_individual] = new_fitness
                    new_population[new_individual] = new_individual
            if np.any(self.fitnesses > self.fitnesses.max() + 1e-6):
                break
            self.population = new_population