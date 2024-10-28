import numpy as np
import random
from scipy.optimize import minimize

class AdaptiveBlackBoxOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.func = None
        self.search_space = None
        self.sample_size = None
        self.sample_indices = None
        self.local_search = False
        self.best_individual = None
        self.best_fitness = float('inf')

    def __call__(self, func):
        if self.func is None:
            self.func = func
            self.search_space = np.random.uniform(-5.0, 5.0, self.dim)
            self.sample_size = 1
            self.sample_indices = None

        if self.budget <= 0:
            raise ValueError("Budget is less than or equal to zero")

        for _ in range(self.budget):
            if self.sample_indices is None:
                self.sample_indices = np.random.choice(self.search_space, size=self.sample_size, replace=False)
            else:
                self.sample_indices = np.random.choice(self.sample_indices, size=self.sample_size, replace=False)
            self.local_search = False

            if self.local_search:
                best_individual = None
                best_fitness = float('inf')
                for i in range(self.sample_size):
                    new_individual = func(self.sample_indices[i])
                    new_fitness = new_individual(self.func(self.sample_indices[i]))
                    if new_fitness < best_fitness:
                        best_individual = new_individual
                        best_fitness = new_fitness
                self.sample_indices = best_individual
                self.local_search = False

            if self.sample_indices is None:
                best_individual = None
                best_fitness = float('inf')
                for i in range(self.sample_size):
                    new_individual = func(self.sample_indices[i])
                    new_fitness = new_individual(self.func(self.sample_indices[i]))
                    if new_fitness < best_fitness:
                        best_individual = new_individual
                        best_fitness = new_fitness
                self.sample_indices = best_individual
                self.local_search = False

            if np.abs(best_individual - func(self.sample_indices)) < 1e-6:
                break

        self.best_individual = self.sample_indices
        self.best_fitness = best_fitness
        return func(self.best_individual)

# One-line description
# Adaptive Black Box Optimization using Adaptive Random Search and Local Search

# Code