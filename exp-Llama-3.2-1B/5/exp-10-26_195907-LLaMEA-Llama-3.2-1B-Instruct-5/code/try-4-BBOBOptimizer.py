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
            # Select a new individual using a novel strategy
            if self.budget <= 0:
                return self.func(self.search_space)
            else:
                # Refine the strategy with a probability of 0.05
                if random.random() < 0.05:
                    # Increase the budget to 2 times the current budget
                    self.budget *= 2
                # Decrease the search space to 1/2 of its current size
                self.search_space = np.vstack((self.search_space, self.search_space[0]))
                # Remove the oldest individual from the search space
                self.search_space = np.delete(self.search_space, 0, axis=0)
                # Evaluate the new individual using the refined strategy
                new_individual = self.evaluate_fitness(new_individual)
                # Replace the old individual with the new one
                self.search_space = np.vstack((self.search_space, new_individual))
                self.search_space = np.delete(self.search_space, 0, axis=0)