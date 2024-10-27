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
        """Randomly change a single element in the individual"""
        idx = random.randint(0, self.dim - 1)
        new_individual = individual.copy()
        new_individual[idx] += random.uniform(-1.0, 1.0)
        return new_individual

    def crossover(self, parent1, parent2):
        """Combine two parents to create a child"""
        child = parent1.copy()
        for i in range(self.dim):
            if random.random() < 0.5:
                child[i] = parent2[i]
        return child

    def selection(self, population):
        """Select the fittest individuals"""
        fitnesses = self.fitnesses
        selected = np.argsort(fitnesses)[-self.population_size:]
        return selected, fitnesses[selected]

# Description: Evolutionary algorithm for black box optimization
# Code: 