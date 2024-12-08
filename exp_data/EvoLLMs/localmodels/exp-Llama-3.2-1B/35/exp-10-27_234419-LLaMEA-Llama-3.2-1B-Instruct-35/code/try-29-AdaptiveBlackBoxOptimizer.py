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
        self.learning_rate = 0.01
        self.momentum = 0.9

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
                if self.sample_indices.size == 1:
                    best_func = func(self.sample_indices)
                else:
                    best_func = np.min([func(self.sample_indices_i) for self.sample_indices_i in self.sample_indices])

                if np.abs(best_func - func(self.sample_indices)) < np.abs(func(self.sample_indices) - func(self.sample_indices)):
                    self.sample_indices = None
                    self.local_search = False
                    self.sample_indices = np.random.choice(self.search_space, size=self.sample_size, replace=False)
                    self.sample_indices = self.sample_indices[:self.sample_size]
                else:
                    self.sample_indices = None
                    self.local_search = False

            if self.sample_indices is None:
                if np.abs(best_func - func(self.sample_indices)) < 1e-6:
                    break

            # Gradient Descent
            gradient = np.gradient(func(self.sample_indices), self.dim)
            self.learning_rate *= 0.9
            self.func = func(self.sample_indices - self.learning_rate * gradient)

            # Momentum
            if self.momentum > 0:
                self.func = func(self.sample_indices + self.momentum * (self.sample_indices - self.func))

            # Normalize
            self.func = self.func / np.max(np.abs(self.func))

            if np.abs(best_func - func(self.sample_indices)) < 1e-6:
                break

        return func(self.sample_indices)