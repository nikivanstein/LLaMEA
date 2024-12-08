import random
import numpy as np

class BBOBOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.search_space = 2 * np.random.uniform(-5.0, 5.0, size=(dim, 2))
        self.func = lambda x: np.sum(x)

    def __call__(self, func, budget=100):
        while True:
            for _ in range(budget):
                x = self.search_space[np.random.randint(0, self.search_space.shape[0])]
                if np.linalg.norm(func(x)) < self.budget / 2:
                    return x
            x = self.search_space[np.random.randint(0, self.search_space.shape[0])]
            self.search_space = np.vstack((self.search_space, x))
            self.search_space = np.delete(self.search_space, 0, axis=0)
            self.search_space = self.search_space[:self.budget]
            self.search_space = self.search_space[np.random.choice(self.search_space.shape[0], size=(self.dim, 2), p=self.search_space / self.budget)]

class NovelMetaheuristicOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.search_space = 2 * np.random.uniform(-5.0, 5.0, size=(dim, 2))
        self.func = lambda x: np.sum(x)

    def __call__(self, func, budget=100):
        optimizer = BBOBOptimizer(budget, self.dim)
        return optimizer(func, budget)

# Example usage:
optimizer = NovelMetaheuristicOptimizer(100, 10)
print(optimizer(func=lambda x: np.sum(x), budget=100))