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
        self.population_size = 100
        self.population_history = []

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
                best_func = func(self.sample_indices)
                if np.abs(best_func - func(self.sample_indices)) < np.abs(func(self.sample_indices) - func(self.sample_indices)):
                    self.sample_indices = None
                    self.local_search = False
                    self.sample_indices = np.random.choice(self.search_space, size=self.sample_size, replace=False)
                    self.sample_indices = self.sample_indices[:self.sample_size]
                else:
                    self.sample_indices = None
                    self.local_search = False

            if self.sample_indices is None:
                best_func = func(self.sample_indices)
                self.sample_indices = None
                self.local_search = False

            if np.abs(best_func - func(self.sample_indices)) < 1e-6:
                break

        return func(self.sample_indices)

    def mutate(self, individual):
        if random.random() < 0.5:
            self.sample_indices = np.random.choice(self.search_space, size=self.sample_size, replace=False)
            self.sample_indices = self.sample_indices[:self.sample_size]
        return individual

    def evolve(self, new_individual):
        self.population_history.append((new_individual, self.__call__(new_individual)))
        if len(self.population_history) > self.population_size:
            self.population_history.pop(0)
        return self.__call__(new_individual)

    def __str__(self):
        return f"AdaptiveBlackBoxOptimizer(budget={self.budget}, dim={self.dim}, population_size={self.population_size})"

# One-line description: Adaptive Black Box Optimization using Adaptive Random Search and Evolutionary Strategies
# Code: 