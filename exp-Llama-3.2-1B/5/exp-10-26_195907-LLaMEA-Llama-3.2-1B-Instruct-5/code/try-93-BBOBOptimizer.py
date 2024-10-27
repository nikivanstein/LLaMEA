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
            new_individual = self.evaluate_fitness(func)
            if np.linalg.norm(new_individual - self.search_space) < 0.05 * np.linalg.norm(self.search_space):
                return new_individual
            self.search_space = np.vstack((self.search_space, new_individual))
            self.search_space = np.delete(self.search_space, 0, axis=0)

    def evaluate_fitness(self, func):
        while True:
            x = self.search_space[np.random.randint(0, self.search_space.shape[0])]
            if np.linalg.norm(func(x)) < self.budget / 2:
                return x
            self.search_space = np.vstack((self.search_space, x))
            self.search_space = np.delete(self.search_space, 0, axis=0)

class NovelMetaheuristicOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.search_space = 2 * np.random.uniform(-5.0, 5.0, size=(dim, 2))
        self.func = lambda x: np.sum(x)

    def __call__(self, func):
        while True:
            new_individual = self.evaluate_fitness(func)
            if np.linalg.norm(new_individual - self.search_space) < 0.05 * np.linalg.norm(self.search_space):
                return new_individual
            self.search_space = np.vstack((self.search_space, new_individual))
            self.search_space = np.delete(self.search_space, 0, axis=0)

# Initialize the optimizer
optimizer = NovelMetaheuristicOptimizer(100, 10)

# Define the function to optimize
def func(x):
    return np.sum(x)

# Evaluate the function 100 times
for _ in range(100):
    print("Evaluating function:", func(np.random.rand(10)))