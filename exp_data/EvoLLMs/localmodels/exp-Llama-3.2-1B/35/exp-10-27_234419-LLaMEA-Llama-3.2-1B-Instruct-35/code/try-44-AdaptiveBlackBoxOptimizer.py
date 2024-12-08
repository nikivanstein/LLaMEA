import numpy as np

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
                for individual in self.sample_indices:
                    fitness = self.func(individual)
                    if fitness < best_fitness:
                        best_individual = individual
                        best_fitness = fitness
                self.sample_indices = None
                self.local_search = False
                self.best_individual = best_individual
                self.best_fitness = best_fitness

            if np.abs(best_individual - self.best_individual) < 1e-6:
                break

        return self.func(self.sample_indices)

# Description: Adaptive Black Box Optimization using Adaptive Random Sampling and Local Search
# Code: 
# ```python
# AdaptiveBlackBoxOptimizer: Adaptive Black Box Optimization using Adaptive Random Sampling and Local Search
# ```python