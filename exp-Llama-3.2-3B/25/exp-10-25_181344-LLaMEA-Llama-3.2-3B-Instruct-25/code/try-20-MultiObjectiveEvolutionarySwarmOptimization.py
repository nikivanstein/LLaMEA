import numpy as np
import random

class MultiObjectiveEvolutionarySwarmOptimization:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 100
        self.candidates = np.random.uniform(-5.0, 5.0, (self.population_size, self.dim))
        self.best_candidates = np.random.uniform(-5.0, 5.0, (self.population_size, self.dim))
        self.fitness_values = np.inf * np.ones((self.population_size,))

    def __call__(self, func):
        for _ in range(self.budget):
            fitness_values = func(self.candidates)
            for i in range(self.population_size):
                self.fitness_values[i] = np.min(fitness_values[i, :])

            # Evolutionary Strategy
            self.candidates[np.random.choice(self.population_size, size=10, replace=False), :] = self.candidates[np.random.choice(self.population_size, size=10, replace=False), :] + \
                                                                                      self.candidates[np.random.choice(self.population_size, size=10, replace=False), :] * \
                                                                                      np.random.uniform(-0.1, 0.1, size=(10, self.dim))

            # Swarm Intelligence
            for _ in range(10):
                new_candidate = np.random.uniform(-5.0, 5.0, self.dim)
                new_fitness_values = func(new_candidate)
                for i in range(self.population_size):
                    new_fitness_values[i] = np.min(new_fitness_values[i, :])
                    if new_fitness_values[i] < self.fitness_values[i]:
                        self.fitness_values[i] = new_fitness_values[i]
                        self.best_candidates[i] = new_candidate

            # Selection
            self.candidates = self.candidates[np.argsort(self.fitness_values, axis=1)]
            self.population_size = self.population_size // 2

            # Mutation
            self.candidates[np.random.choice(self.population_size, size=self.population_size, replace=False), :] += np.random.uniform(-0.1, 0.1, size=(self.population_size, self.dim))

            # Check if the best candidate is improved
            for i in range(self.population_size):
                if self.fitness_values[i] < np.min(func(self.best_candidates[i, :])):
                    self.candidates[i, :] = self.best_candidates[i, :]

        return self.best_candidates, np.min(func(self.best_candidates[:, 0])), np.min(func(self.best_candidates[:, 1]))

# Example usage:
def func(x):
    return np.array([x[0]**2 + x[1]**2, x[0]**2 + x[2]**2])

moe_SSO = MultiObjectiveEvolutionarySwarmOptimization(budget=100, dim=3)
best_candidates, best_fitness_1, best_fitness_2 = moe_SSO(func)
print(f"Best candidates: {best_candidates}, Best fitness 1: {best_fitness_1}, Best fitness 2: {best_fitness_2}")