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

    def mutate(self, func):
        for _ in range(self.budget // 2):
            x = self.search_space[np.random.randint(0, self.search_space.shape[0])]
            y = x + np.random.uniform(-1.0, 1.0)
            if np.linalg.norm(func(x)) < self.budget / 2 and np.linalg.norm(func(y)) < self.budget / 2:
                self.search_space = np.delete(self.search_space, 0, axis=0)
                self.search_space = np.vstack((self.search_space, [x, y]))
                self.search_space = np.delete(self.search_space, 0, axis=0)
                break
        return self.search_space

    def evolve(self, func, mutation_rate):
        new_individuals = []
        for _ in range(self.budget):
            new_individual = self.mutate(func)
            if np.random.rand() < mutation_rate:
                new_individual = self.mutate(func)
            new_individuals.append(new_individual)
        return new_individuals