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
            for _ in range(self.budget):
                x = self.search_space[np.random.randint(0, self.search_space.shape[0])]
                if np.linalg.norm(func(x)) < self.budget / 2:
                    return x
            x = self.search_space[np.random.randint(0, self.search_space.shape[0])]
            self.search_space = np.vstack((self.search_space, x))
            self.search_space = np.delete(self.search_space, 0, axis=0)

            # Refine the strategy using the following two rules
            # Rule 1: Change the individual lines of the selected solution to refine its strategy
            # Rule 2: Apply a probability 0.05 to change the individual lines of the selected solution
            if random.random() < 0.05:
                rule1 = random.choice(['line1', 'line2'])
                if rule1 == 'line1':
                    self.search_space[:, 0] = np.random.uniform(-1.0, 1.0)
                elif rule1 == 'line2':
                    self.search_space[:, 1] = np.random.uniform(-1.0, 1.0)

            if random.random() < 0.05:
                rule2 = random.choice(['line1', 'line2'])
                if rule2 == 'line1':
                    self.search_space[:, 0] = np.random.uniform(-0.5, 0.5)
                elif rule2 == 'line2':
                    self.search_space[:, 1] = np.random.uniform(-0.5, 0.5)