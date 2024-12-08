import numpy as np
import random

class HyperHeuristicsForBBO:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 100
        self.candidates = np.random.uniform(-5.0, 5.0, (self.population_size, self.dim))
        self.best_candidate = np.random.uniform(-5.0, 5.0, self.dim)
        self.best_fitness = np.inf
        self.strategies = [np.random.uniform(-0.1, 0.1, size=(10, self.dim)), np.random.uniform(-0.1, 0.1, size=(10, self.dim))]
        self.fitness_history = np.zeros((self.budget, self.population_size))

    def __call__(self, func):
        for _ in range(self.budget):
            fitness = func(self.candidates[:, 0])
            self.best_candidate = self.candidates[np.argmin(self.candidates[:, 0]), :]
            self.best_fitness = fitness

            # Hyper-Heuristics
            for _ in range(10):
                new_strategy = np.copy(self.strategies[np.random.choice(len(self.strategies), size=1, replace=False)])
                new_candidate = self.candidates[np.random.choice(self.population_size, size=1, replace=False), :] + new_strategy[np.random.choice(self.population_size, size=1, replace=False), :]
                new_fitness = func(new_candidate)
                if new_fitness < self.best_fitness:
                    self.best_candidate = new_candidate
                    self.best_fitness = new_fitness
                    self.candidates[np.argmin(self.candidates[:, 0]), :] = new_candidate

            # Hyper-Evolutionary Strategy
            for _ in range(10):
                new_strategy = np.copy(self.strategies[np.random.choice(len(self.strategies), size=1, replace=False)])
                new_strategy += np.random.uniform(-0.1, 0.1, size=(10, self.dim))
                new_strategy = self.strategies[np.argmin(np.sum(new_strategy**2, axis=1))]
                new_candidate = self.candidates[np.random.choice(self.population_size, size=1, replace=False), :] + new_strategy[np.random.choice(self.population_size, size=1, replace=False), :]
                new_fitness = func(new_candidate)
                if new_fitness < self.best_fitness:
                    self.best_candidate = new_candidate
                    self.best_fitness = new_fitness
                    self.candidates[np.argmin(self.candidates[:, 0]), :] = new_candidate

            # Selection
            self.candidates = self.candidates[np.argsort(self.candidates[:, 0])]
            self.population_size = self.population_size // 2

            # Mutation
            self.candidates[np.random.choice(self.population_size, size=self.population_size, replace=True), :] += np.random.uniform(-0.1, 0.1, size=(self.population_size, self.dim))

            # Check if the best candidate is improved
            if self.best_fitness < func(self.best_candidate):
                self.candidates[np.argmin(self.candidates[:, 0]), :] = self.best_candidate

            # Record fitness history
            self.fitness_history[_] = self.candidates[:, 0]

        return self.best_candidate, self.best_fitness

# Example usage:
def func(x):
    return x[0]**2 + x[1]**2

hyper_HHSO = HyperHeuristicsForBBO(budget=100, dim=2)
best_candidate, best_fitness = hyper_HHSO(func)
print(f"Best candidate: {best_candidate}, Best fitness: {best_fitness}")