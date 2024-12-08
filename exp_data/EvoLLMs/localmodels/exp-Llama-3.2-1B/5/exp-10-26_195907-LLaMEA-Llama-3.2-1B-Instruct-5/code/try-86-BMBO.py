import random
import numpy as np

class BMBO:
    def __init__(self, budget, dim, n_iter):
        self.budget = budget
        self.dim = dim
        self.search_space = 2 * np.random.uniform(-5.0, 5.0, size=(dim, n_iter * dim))
        self.func = lambda x: np.sum(x)
        self.n_iter = n_iter

    def __call__(self, func):
        for _ in range(self.budget):
            for _ in range(self.n_iter):
                x = self.search_space[np.random.randint(0, self.search_space.shape[0])]
                if np.linalg.norm(func(x)) < self.budget / 2:
                    return x
            self.search_space = np.vstack((self.search_space, x))
            self.search_space = np.delete(self.search_space, 0, axis=0)

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

class BMBOOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.search_space = 2 * np.random.uniform(-5.0, 5.0, size=(dim, 2))
        self.func = lambda x: np.sum(x)
        self.n_iter = 100

    def __call__(self, func):
        for _ in range(self.budget):
            for _ in range(self.n_iter):
                x = self.search_space[np.random.randint(0, self.search_space.shape[0])]
                if np.linalg.norm(func(x)) < self.budget / 2:
                    return x
            self.search_space = np.vstack((self.search_space, x))
            self.search_space = np.delete(self.search_space, 0, axis=0)

# Description: Novel Metaheuristic Algorithm for Black Box Optimization
# Code: 