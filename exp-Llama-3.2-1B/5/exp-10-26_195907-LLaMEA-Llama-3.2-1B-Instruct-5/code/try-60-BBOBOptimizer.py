import random
import numpy as np

class BBOBOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.search_space = 2 * np.random.uniform(-5.0, 5.0, size=(dim, 2))
        self.func = lambda x: np.sum(x)

    def __call__(self, func, mutation_rate=0.01, crossover_rate=0.5):
        """
        Optimizes the black box function using the given mutation and crossover rates.

        Parameters:
        func (function): The black box function to optimize.
        mutation_rate (float, optional): The probability of mutation in the offspring. Defaults to 0.01.
        crossover_rate (float, optional): The probability of crossover in the offspring. Defaults to 0.5.

        Returns:
        tuple: The optimized individual and its fitness.
        """
        while True:
            for _ in range(self.budget):
                x = self.search_space[np.random.randint(0, self.search_space.shape[0])]
                if np.linalg.norm(func(x)) < self.budget / 2:
                    return x, func(x)
            x = self.search_space[np.random.randint(0, self.search_space.shape[0])]
            self.search_space = np.vstack((self.search_space, x))
            self.search_space = np.delete(self.search_space, 0, axis=0)

            # Refine the strategy by changing the individual lines of the selected solution
            if random.random() < 0.05:
                x[:, 0] *= 1 + random.uniform(-0.1, 0.1)
            if random.random() < 0.05:
                x[:, 1] *= 1 + random.uniform(-0.1, 0.1)

            # Ensure the individual stays within the search space
            x = np.clip(x, self.search_space[:, 0], self.search_space[:, 1])

# Example usage:
optimizer = BBOBOptimizer(100, 10)
optimizer.__call__(optimizer.func)