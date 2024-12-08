import numpy as np
import random
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
                x = np.array([self.search_space[np.random.randint(0, self.search_space.shape[0])]]).T
                if np.linalg.norm(func(x)) < self.budget / 2:
                    return x
            x = np.array([self.search_space[np.random.randint(0, self.search_space.shape[0])]]).T
            self.search_space = np.vstack((self.search_space, x))
            self.search_space = np.delete(self.search_space, 0, axis=0)
            # Refine strategy by changing the individual lines of the selected solution
            if np.random.rand() < 0.05:
                x[0] = self.search_space[0, 0] + random.uniform(-0.1, 0.1)
            elif np.random.rand() < 0.05:
                x[0] = self.search_space[0, 0] - random.uniform(-0.1, 0.1)
            else:
                x[0] = self.search_space[0, 0] + random.uniform(-0.1, 0.1) * 2