# Description: Novel Metaheuristic Algorithm for Black Box Optimization
# Code: 
# ```python
import random
import numpy as np

class BBOBOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.search_space = 2 * np.random.uniform(-5.0, 5.0, size=(dim, 2))
        self.func = lambda x: np.sum(x)

    def __call__(self, func):
        while True:
            for _ in range(min(self.budget, self.dim)):
                x = self.search_space[np.random.randint(0, self.search_space.shape[0])]
                if np.linalg.norm(func(x)) < self.budget / 2:
                    return x
            x = self.search_space[np.random.randint(0, self.search_space.shape[0])]
            self.search_space = np.vstack((self.search_space, x))
            self.search_space = np.delete(self.search_space, 0, axis=0)
            if np.random.rand() < 0.05:  # Refine strategy with probability 0.05
                x = self.search_space[np.random.randint(0, self.search_space.shape[0])]
            self.search_space = np.delete(self.search_space, 0, axis=0)

class NovelMetaheuristicOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.search_space = 2 * np.random.uniform(-5.0, 5.0, size=(dim, 2))
        self.func = lambda x: np.sum(x)

    def __call__(self, func):
        optimizer = BBOBOptimizer(self.budget, self.dim)
        return optimizer(func)

# Example usage:
optimizer = NovelMetaheuristicOptimizer(100, 10)
func = lambda x: np.sum(x)
print(optimizer(func))  # Output: a refined individual