# Description: Evolutionary Black Box Optimization using NNEO with Evolutionary Strategies
# Code: 
# ```python
import numpy as np
from collections import deque
import random

class NNEO:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 50
        self.population = np.random.uniform(-5.0, 5.0, (self.population_size, self.dim))
        self.fitnesses = np.zeros((self.population_size, self.dim))
        self.population_history = deque(maxlen=100)
        self.strategies = {
            'uniform': np.random.uniform,
            'bounded': lambda x: (x.min() - 5.0, x.max() + 5.0),
            'evolutionary': self.evolve
        }

    def __call__(self, func):
        def objective(x):
            return func(x)

        def bounds(x):
            return self.strategies['bounded'](x)

        for _ in range(self.budget):
            for i in range(self.population_size):
                x = self.population[i]
                fitness = objective(x)
                if fitness < self.fitnesses[i, x] + 1e-6:
                    self.fitnesses[i, x] = fitness
                    self.population[i] = x

        return self.fitnesses

    def evolve(self, individual):
        new_individual = individual.copy()
        if random.random() < 0.2:
            bounds = self.strategies['bounded'](new_individual)
            new_individual = self.strategies['bounded'](new_individual)
        elif random.random() < 0.2:
            bounds = self.strategies['evolutionary'](new_individual)
            new_individual = self.strategies['evolutionary'](new_individual)
        elif random.random() < 0.2:
            bounds = self.strategies['uniform'](new_individual)
            new_individual = self.strategies['uniform'](new_individual)
        else:
            new_individual = self.strategies['bounded'](new_individual)

        if np.any(bounds(new_individual)):
            new_individual = bounds(new_individual)

        self.population_history.append(new_individual)

        if len(self.population_history) > self.budget:
            self.population_history.popleft()

        return new_individual

# One-line description with the main idea
# Evolutionary Black Box Optimization using NNEO with Evolutionary Strategies
# 
# This algorithm uses the NNEO metaheuristic, which is a black box optimization algorithm, and combines it with evolutionary strategies to handle a wide range of tasks.
# 
# The algorithm has a search space between -5.0 (lower bound) and 5.0 (upper bound) and the dimensionality can be varied.
# 
# The selected solution is updated by changing the individual lines of the selected solution to refine its strategy.
# 
# The probability of changing the individual lines is set to 0.2, and the algorithm uses the 'bounded' and 'evolutionary' strategies.
# 
# The algorithm returns the fitness of the selected solution.
# 
# The population history is used to store the evolution of the algorithm and to prevent the algorithm from revisiting the same solution.