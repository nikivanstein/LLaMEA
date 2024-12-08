import numpy as np
import random
from collections import deque
import copy

class HyperCanny:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.func_evals = 0
        self.population_size = 100
        self.population = self.initialize_population()

    def initialize_population(self):
        # Initialize the population with random solutions
        return [copy.deepcopy(np.random.uniform(-5.0, 5.0, self.dim)) for _ in range(self.population_size)]

    def __call__(self, func):
        # Evaluate the function for each individual in the population
        while self.func_evals < self.budget:
            # Grid search
            x_values = np.linspace(-5.0, 5.0, 100)
            y_values = func(x_values)
            grid = dict(zip(x_values, y_values))
            best_x, best_y = None, None
            for x, y in grid.items():
                if x < best_x or (x == best_x and y < best_y):
                    best_x, best_y = x, y
            # Random search
            random_x_values = np.random.uniform(-5.0, 5.0, self.dim)
            random_y_values = func(random_x_values)
            random_x_values = np.array([x for x, y in zip(random_x_values, random_y_values) if -5.0 <= x <= 5.0])
            random_y_values = np.array([y for x, y in zip(random_x_values, random_y_values) if -5.0 <= y <= 5.0])
            # Evolutionary algorithm
            self.func_evals += 1
            x_values = random_x_values
            y_values = random_y_values
            for _ in range(100):
                x_new = x_values + np.random.uniform(-0.1, 0.1, self.dim)
                y_new = y_values + np.random.uniform(-0.1, 0.1, self.dim)
                if -5.0 <= x_new <= 5.0 and -5.0 <= y_new <= 5.0:
                    x_values = x_new
                    y_values = y_new
                    break
            # Check if the new solution is better
            if np.max(y_values) > np.max(y_values + 0.1):
                best_x, best_y = x_values, y_values
        return best_x, best_y

    def mutate(self, individual):
        # Randomly mutate an individual in the population
        if random.random() < 0.1:
            x = random.uniform(-5.0, 5.0)
            y = random.uniform(-5.0, 5.0)
            individual[x], individual[y] = individual[y], individual[x]
        return individual

    def crossover(self, parent1, parent2):
        # Perform crossover between two parents to create a new individual
        child = np.copy(parent1)
        for i in range(self.dim):
            if random.random() < 0.5:
                child[i] = parent2[i]
        return child

    def select(self, parent1, parent2):
        # Select the fittest parent to reproduce
        if random.random() < 0.1:
            return parent1
        else:
            return parent2

# One-line description:
# HyperCanny: A novel metaheuristic algorithm for solving black box optimization problems using a combination of grid search, random search, and evolutionary algorithms.