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
            # Select a new individual based on the budget
            if random.random() < 0.05:
                # Increase the dimensionality to explore more areas
                self.dim += 1
                self.search_space = 2 * np.random.uniform(-5.0, 5.0, size=(self.dim, 2))
            else:
                # Select an individual from the current search space
                new_individual = self.search_space[np.random.randint(0, self.search_space.shape[0])]
                if np.linalg.norm(func(new_individual)) < self.budget / 2:
                    return new_individual
            # Update the search space with the new individual
            self.search_space = np.vstack((self.search_space, new_individual))
            self.search_space = np.delete(self.search_space, 0, axis=0)

# Novel Metaheuristic Algorithm for Black Box Optimization
# Description: Novel Metaheuristic Algorithm for Black Box Optimization
# Code: 