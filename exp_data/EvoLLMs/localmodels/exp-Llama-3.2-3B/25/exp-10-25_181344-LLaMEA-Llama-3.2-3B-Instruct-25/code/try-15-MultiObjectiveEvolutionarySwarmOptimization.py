import numpy as np
import random

class MultiObjectiveEvolutionarySwarmOptimization:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 100
        self.candidates = np.random.uniform(-5.0, 5.0, (self.population_size, self.dim))
        self.best_candidates = np.random.uniform(-5.0, 5.0, (self.population_size, self.dim))
        self.best_fitness = np.inf

    def __call__(self, func):
        for _ in range(self.budget):
            fitness = func(self.candidates[:, 0])
            self.best_candidates = self.candidates[np.argmin(fitness), :]
            self.best_fitness = fitness

            # Evolutionary Strategy
            self.candidates[np.random.choice(self.population_size, size=10, replace=True), :] = self.candidates[np.random.choice(self.population_size, size=10, replace=True), :] + \
                                                                                      self.candidates[np.random.choice(self.population_size, size=10, replace=True), :] * \
                                                                                      np.random.uniform(-0.1, 0.1, size=(10, self.dim))

            # Swarm Intelligence
            for _ in range(10):
                new_candidates = np.random.uniform(-5.0, 5.0, (self.population_size, self.dim))
                new_fitness = func(new_candidates)
                new_indices = np.argsort(new_fitness)
                new_candidates = new_candidates[new_indices]

                for i in range(self.population_size):
                    if np.random.rand() < 0.25:
                        new_candidates[i, :] += np.random.uniform(-0.1, 0.1, size=(1, self.dim))

                new_fitness = func(new_candidates)
                new_indices = np.argsort(new_fitness)
                new_candidates = new_candidates[new_indices]

                if np.any(new_fitness < self.best_fitness):
                    self.best_candidates = new_candidates[np.argmin(new_fitness)]
                    self.best_fitness = np.min(new_fitness)
                    self.candidates[np.argmin(fitness), :] = self.best_candidates

            # Selection
            self.candidates = self.candidates[np.argsort(fitness)]
            self.population_size = self.population_size // 2

            # Mutation
            self.candidates[np.random.choice(self.population_size, size=self.population_size, replace=True), :] += np.random.uniform(-0.1, 0.1, size=(self.population_size, self.dim))

            # Check if the best candidate is improved
            if np.any(self.best_fitness < func(self.best_candidates)):
                self.candidates[np.argmin(fitness), :] = self.best_candidates

        return self.best_candidates, self.best_fitness

# Example usage:
def func(x):
    return x[0]**2 + x[1]**2

multi_objective_ESO = MultiObjectiveEvolutionarySwarmOptimization(budget=100, dim=2)
best_candidates, best_fitness = multi_objective_ESO(func)
print(f"Best candidates: {best_candidates}, Best fitness: {best_fitness}")