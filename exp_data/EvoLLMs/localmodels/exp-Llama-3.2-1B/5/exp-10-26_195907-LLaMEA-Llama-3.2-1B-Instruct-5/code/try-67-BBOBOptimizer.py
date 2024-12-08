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
                if random.random() < 0.05:
                    x = self.search_space[np.random.randint(0, self.search_space.shape[0])]
                    if np.linalg.norm(func(x)) < self.budget / 2:
                        return x
                elif random.random() < 0.05:
                    x = self.search_space[np.random.randint(0, self.search_space.shape[0])]
                    if np.linalg.norm(func(x)) < self.budget / 2:
                        return x
            x = self.search_space[np.random.randint(0, self.search_space.shape[0])]
            self.search_space = np.vstack((self.search_space, x))
            self.search_space = np.delete(self.search_space, 0, axis=0)

class BBOBMetaheuristic:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.search_space = 2 * np.random.uniform(-5.0, 5.0, size=(dim, 2))
        self.func = lambda x: np.sum(x)
        self.population = [BBOBOptimizer(budget, dim) for _ in range(100)]

    def __call__(self, func):
        best_individual = None
        best_fitness = -np.inf
        for _ in range(100):
            for individual in self.population:
                fitness = individual.func(func)
                if fitness > best_fitness:
                    best_individual = individual
                    best_fitness = fitness
            if random.random() < 0.05:
                new_individual = BBOBOptimizer(self.budget, self.dim)
                best_individual = BBOBOptimizer(self.budget, self.dim)
                new_fitness = new_individual.func(func)
                if new_fitness > best_fitness:
                    best_individual = new_individual
                    best_fitness = new_fitness
            if np.linalg.norm(func(best_individual.func)) < self.budget / 2:
                break
        return best_individual

optimizer = BBOBMetaheuristic(budget=100, dim=10)
best_individual = optimizer(__call__)
print("Best Individual:", best_individual.func(best_individual.func))
print("Best Fitness:", np.linalg.norm(best_individual.func(best_individual.func)))