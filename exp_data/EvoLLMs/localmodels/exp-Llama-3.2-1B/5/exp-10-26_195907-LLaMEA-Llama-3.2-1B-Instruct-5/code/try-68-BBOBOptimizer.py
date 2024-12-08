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

    def __call__(self, func, budget):
        while True:
            for _ in range(budget):
                if np.linalg.norm(func(self.search_space[np.random.randint(0, self.search_space.shape[0])])) >= self.budget / 2:
                    return self.search_space[np.random.randint(0, self.search_space.shape[0])]
                x = self.search_space[np.random.randint(0, self.search_space.shape[0])]
                if np.linalg.norm(func(x)) < self.budget / 2:
                    self.search_space = np.vstack((self.search_space, x))
                    self.search_space = np.delete(self.search_space, 0, axis=0)
            x = self.search_space[np.random.randint(0, self.search_space.shape[0])]
            if np.linalg.norm(func(x)) < self.budget / 2:
                return x