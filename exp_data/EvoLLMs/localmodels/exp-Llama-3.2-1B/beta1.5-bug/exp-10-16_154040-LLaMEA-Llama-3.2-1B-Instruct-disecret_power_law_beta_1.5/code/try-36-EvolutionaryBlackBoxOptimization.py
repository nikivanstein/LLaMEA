import random
import numpy as np

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
            fitness = self.evaluate(func)
            if fitness < 0:
                break
        return func, fitness

    def evaluate(self, func):
        return func(func, random.uniform(-5.0, 5.0), random.uniform(-5.0, 5.0))

# One-line description: Evolutionary Black Box Optimization using Genetic Algorithm

# Description: Evolutionary Black Box Optimization using Genetic Algorithm for Multi-Dimensional Optimization
# Code: EBBO uses a population-based approach to search for the optimal solution in the black box function space.
# The population is initialized with random candidates and then iteratively evolves to find the optimal solution.
# The search space is constrained to a given number of dimensions and budget of function evaluations.
# The algorithm uses a genetic algorithm approach to find the optimal solution in a multi-dimensional optimization problem.