import numpy as np
import random

class DifferentialEvolutionWithSwarm:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 100
        self.candidates = np.random.uniform(-5.0, 5.0, (self.population_size, self.dim))
        self.best_candidate = np.random.uniform(-5.0, 5.0, self.dim)
        self.best_fitness = np.inf
        self.swarm_size = 10

    def __call__(self, func):
        for _ in range(self.budget):
            fitness = func(self.candidates[:, 0])
            self.best_candidate = self.candidates[np.argmin(self.candidates[:, 0]), :]
            self.best_fitness = fitness

            # Differential Evolution
            for i in range(self.population_size):
                j = np.random.randint(0, self.population_size)
                k = np.random.randint(0, self.population_size)
                new_candidate = self.candidates[i, :] + \
                                self.candidates[j, :] - \
                                self.candidates[k, :] + \
                                2 * np.random.uniform(-0.1, 0.1, size=self.dim)
                new_fitness = func(new_candidate)
                if new_fitness < self.best_fitness:
                    self.best_candidate = new_candidate
                    self.best_fitness = new_fitness
                    self.candidates[np.argmin(self.candidates[:, 0]), :] = new_candidate

            # Swarm Intelligence
            for _ in range(self.swarm_size):
                new_candidate = np.random.uniform(-5.0, 5.0, self.dim)
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

            # Refine strategy
            for i in range(self.population_size):
                if np.random.rand() < 0.25:
                    self.candidates[i, :] += np.random.uniform(-0.05, 0.05, size=self.dim)

            # Check if the best candidate is improved
            if self.best_fitness < func(self.best_candidate):
                self.candidates[np.argmin(self.candidates[:, 0]), :] = self.best_candidate

        return self.best_candidate, self.best_fitness

# Example usage:
def func(x):
    return x[0]**2 + x[1]**2

de_with_swarm = DifferentialEvolutionWithSwarm(budget=100, dim=2)
best_candidate, best_fitness = de_with_swarm(func)
print(f"Best candidate: {best_candidate}, Best fitness: {best_fitness}")