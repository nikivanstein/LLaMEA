import numpy as np
import random

class MetaSwarm:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 100
        self.candidates = np.random.uniform(-5.0, 5.0, (self.population_size, self.dim))
        self.best_candidate = np.random.uniform(-5.0, 5.0, self.dim)
        self.best_fitness = np.inf
        self.meta_params = np.random.uniform(-0.5, 0.5, (10, self.dim))

    def __call__(self, func):
        for _ in range(self.budget):
            fitness = func(self.candidates[:, 0])
            self.best_candidate = self.candidates[np.argmin(self.candidates[:, 0]), :]
            self.best_fitness = fitness

            # Meta-heuristics
            self.candidates[np.random.choice(self.population_size, size=10, replace=False), :] = self.candidates[np.random.choice(self.population_size, size=10, replace=False), :] + \
                                                                                      self.candidates[np.random.choice(self.population_size, size=10, replace=False), :] * \
                                                                                      np.random.uniform(-0.1, 0.1, size=(10, self.dim))

            # Swarm Intelligence
            for _ in range(10):
                new_candidate = self.candidates[np.random.choice(self.population_size, size=1, replace=False), :]
                new_fitness = func(new_candidate)
                if new_fitness < self.best_fitness:
                    self.best_candidate = new_candidate
                    self.best_fitness = new_fitness
                    self.candidates[np.argmin(self.candidates[:, 0]), :] = new_candidate

            # Selection
            self.candidates = self.candidates[np.argsort(self.candidates[:, 0])]
            self.population_size = self.population_size // 2

            # Mutation
            self.candidates[np.random.choice(self.population_size, size=self.population_size, replace=False), :] += np.random.uniform(-0.1, 0.1, size=(self.population_size, self.dim))

            # Meta-heuristics refinement
            self.meta_params = self.meta_params * 0.75 + np.random.uniform(-0.25, 0.25, size=(10, self.dim))

            # Check if the best candidate is improved
            if self.best_fitness < func(self.best_candidate):
                self.candidates[np.argmin(self.candidates[:, 0]), :] = self.best_candidate

        return self.best_candidate, self.best_fitness

# Example usage:
def func(x):
    return x[0]**2 + x[1]**2

meta_swarm = MetaSwarm(budget=100, dim=2)
best_candidate, best_fitness = meta_swarm(func)
print(f"Best candidate: {best_candidate}, Best fitness: {best_fitness}")