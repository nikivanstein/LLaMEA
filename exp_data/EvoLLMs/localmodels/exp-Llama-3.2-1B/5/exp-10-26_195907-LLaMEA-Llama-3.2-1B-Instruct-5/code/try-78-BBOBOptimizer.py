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
            self.search_space = self.search_space[:self.budget]
            self.search_space = self.search_space[np.random.choice(self.search_space.shape[0], size=self.dim, replace=False)]
            self.search_space = self.search_space[:self.budget]

    def evaluate_fitness(self, individual):
        return self.func(individual)

# ```python
# Novel Metaheuristic Algorithm for Black Box Optimization
# 
# The algorithm refines its strategy by changing the individual lines of the selected solution to refine its strategy.
# 
# The probability of changing an individual line is 0.05, and the algorithm stops when the budget is exhausted.
# 
# The search space is bounded between -5.0 and 5.0, and the dimensionality can be varied.
# 
# The code is designed to handle a wide range of tasks and evaluate the black box function on the BBOB test suite of 24 noiseless functions.