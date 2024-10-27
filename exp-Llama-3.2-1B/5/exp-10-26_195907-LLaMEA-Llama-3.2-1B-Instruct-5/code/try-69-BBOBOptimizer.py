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
            # Select a new individual based on the budget constraint
            # and the probability of refining the individual
            new_individual = self.evaluate_fitness(self.search_space[np.random.randint(0, self.search_space.shape[0])], 
                                                   self.budget / 2)

            # Evaluate the new individual using the original function
            updated_individual = self.evaluate_fitness(new_individual, self.budget)

            # Refine the individual with a probability of 0.05
            if np.random.rand() < 0.05:
                updated_individual = self.search_space[np.random.randint(0, self.search_space.shape[0])]

            # Replace the old individual with the new one
            self.search_space = np.vstack((self.search_space, updated_individual))
            self.search_space = np.delete(self.search_space, 0, axis=0)

            return updated_individual