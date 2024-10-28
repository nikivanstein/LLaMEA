import random
import numpy as np
from scipy.optimize import differential_evolution

class EvolutionaryBlackBoxOptimization:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 50
        self.population = self.initialize_population()

    def initialize_population(self):
        population = []
        for _ in range(self.population_size):
            dim = self.dim * random.random()
            func = self.generate_func(dim)
            population.append((func, random.uniform(-5.0, 5.0), random.uniform(-5.0, 5.0)))
        return population

    def generate_func(self, dim):
        return np.sin(np.sqrt(dim))

    def __call__(self, func):
        for _ in range(self.budget):
            fitness = differential_evolution(self.evaluate, self.population)
            if fitness.fun < 0:
                break
        return func, fitness.fun

    def evaluate(self, func):
        return func(func, random.uniform(-5.0, 5.0), random.uniform(-5.0, 5.0))

# One-line description: Evolutionary Black Box Optimization using Genetic Algorithm

# Code: Evolutionary Black Box Optimization using Genetic Algorithm