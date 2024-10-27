import random
import numpy as np

class BBOBOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.search_space = 2 * np.random.uniform(-5.0, 5.0, size=(dim, 2))
        self.func = lambda x: np.sum(x)

    def __call__(self, func):
        def evaluate_fitness(individual):
            return self.func(individual)

        def next_individual(budget):
            while True:
                for _ in range(min(budget, self.budget)):
                    x = self.search_space[np.random.randint(0, self.search_space.shape[0])]
                    if np.linalg.norm(evaluate_fitness(x)) < self.budget / 2:
                        return x
                x = self.search_space[np.random.randint(0, self.search_space.shape[0])]
                self.search_space = np.vstack((self.search_space, x))
                self.search_space = np.delete(self.search_space, 0, axis=0)

        return next_individual(self.budget)

# Novel Metaheuristic Algorithm for Black Box Optimization
# Code: 