import numpy as np
from scipy.optimize import differential_evolution

class AdaptiveBlackBoxOptimizer:
    def __init__(self, budget, dim, strategy="random"):
        self.budget = budget
        self.dim = dim
        self.func = None
        self.search_space = None
        self.sample_size = None
        self.sample_indices = None
        self.local_search = False
        self.evolutionary_strategy = strategy

    def __call__(self, func):
        if self.func is None:
            self.func = func
            self.search_space = np.random.uniform(-5.0, 5.0, self.dim)
            self.sample_size = 1
            self.sample_indices = None

        if self.budget <= 0:
            raise ValueError("Budget is less than or equal to zero")

        for _ in range(self.budget):
            if self.evolutionary_strategy == "random":
                self.sample_indices = np.random.choice(self.search_space, size=self.sample_size, replace=False)
            elif self.evolutionary_strategy == "local":
                best_func = func(self.sample_indices)
                self.sample_indices = None
                self.local_search = False
                self.sample_indices = np.random.choice(self.search_space, size=self.sample_size, replace=False)
                self.sample_indices = self.sample_indices[:self.sample_size]
            elif self.evolutionary_strategy == "adaptive":
                self.sample_indices = np.random.choice(self.search_space, size=self.sample_size, replace=False)
                best_func = func(self.sample_indices)
                self.sample_indices = None
                self.local_search = False
                if np.abs(best_func - func(self.sample_indices)) < 1e-6:
                    break
            elif self.evolutionary_strategy == "bayes":
                self.sample_indices = np.random.choice(self.search_space, size=self.sample_size, replace=False)
                best_func = func(self.sample_indices)
                self.sample_indices = None
                self.local_search = False
                if np.abs(best_func - func(self.sample_indices)) < 1e-6:
                    break
            elif self.evolutionary_strategy == "genetic":
                self.sample_indices = np.random.choice(self.search_space, size=self.sample_size, replace=False)
                best_func = func(self.sample_indices)
                self.sample_indices = None
                self.local_search = False
                if np.abs(best_func - func(self.sample_indices)) < 1e-6:
                    break

        return func(self.sample_indices)

# Description: Adaptive Black Box Optimization using Adaptive Sampling and Local Search with Evolutionary Strategies
# Code: 