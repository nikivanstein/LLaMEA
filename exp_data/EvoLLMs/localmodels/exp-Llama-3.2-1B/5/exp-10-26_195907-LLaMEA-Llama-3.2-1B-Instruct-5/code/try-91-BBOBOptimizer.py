import random
import numpy as np
from scipy.optimize import minimize

class BBOBOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.search_space = 2 * np.random.uniform(-5.0, 5.0, size=(dim, 2))
        self.func = lambda x: np.sum(x)

    def __call__(self, func):
        while True:
            for _ in range(self.budget):
                x = self.search_space[np.random.randint(0, self.search_space.shape[0])]
                if np.linalg.norm(func(x)) < self.budget / 2:
                    return x
            x = self.search_space[np.random.randint(0, self.search_space.shape[0])]
            self.search_space = np.vstack((self.search_space, x))
            self.search_space = np.delete(self.search_space, 0, axis=0)

    def evolve(self, func):
        """Refine the solution using a probabilistic strategy."""
        new_individuals = []
        for _ in range(self.budget):
            new_individual = self.evaluate_fitness(func)
            if np.random.rand() < 0.05:  # Refine strategy with 5% probability
                new_individual = self.refine_strategy(func, new_individual)
            new_individuals.append(new_individual)
        return new_individuals

    def refine_strategy(self, func, individual):
        """Refine the individual using a greedy strategy."""
        return individual + np.random.uniform(-5.0, 5.0, size=self.dim)

    def evaluate_fitness(self, func):
        """Evaluate the fitness of an individual using the given function."""
        return func(self.evaluate_individual(func))

    def evaluate_individual(self, func):
        """Evaluate the fitness of an individual using the given function."""
        return func(self.search_space[np.random.randint(0, self.search_space.shape[0])])

# Description: Novel Metaheuristic Algorithm for Black Box Optimization
# Code: 