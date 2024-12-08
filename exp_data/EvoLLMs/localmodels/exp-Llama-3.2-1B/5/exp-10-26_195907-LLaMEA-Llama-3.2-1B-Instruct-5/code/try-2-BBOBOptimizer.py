import random
import numpy as np
import math

class BBOBOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.search_space = 2 * np.random.uniform(-5.0, 5.0, size=(dim, 2))
        self.func = lambda x: np.sum(x)

    def __call__(self, func):
        while True:
            new_individual = self.evaluate_fitness(self.func)
            old_individual = self.func(new_individual)
            delta = old_individual - new_individual

            if random.random() < 0.05:
                new_individual = self.search_space[np.random.randint(0, self.search_space.shape[0])]

            if delta < 0:
                self.search_space = np.delete(self.search_space, 0, axis=0)
            elif delta > 0:
                new_individual = self.search_space[np.random.randint(0, self.search_space.shape[0])]

            self.search_space = np.vstack((self.search_space, new_individual))
            self.search_space = np.delete(self.search_space, 0, axis=0)

            if math.exp(-self.budget / (delta ** 2)) > random.random():
                return new_individual