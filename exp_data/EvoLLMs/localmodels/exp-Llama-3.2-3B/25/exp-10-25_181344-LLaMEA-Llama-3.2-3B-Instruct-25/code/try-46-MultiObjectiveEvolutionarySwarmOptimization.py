import numpy as np
import random

class MultiObjectiveEvolutionarySwarmOptimization:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 100
        self.candidates = np.random.uniform(-5.0, 5.0, (self.population_size, self.dim))
        self.best_candidates = np.random.uniform(-5.0, 5.0, (self.population_size, self.dim))
        self.fitnesses = np.inf * np.ones((self.population_size, 2))
        self.best_fitness = np.inf

    def __call__(self, func):
        for _ in range(self.budget):
            fitness = func(self.candidates[:, 0])
            self.fitnesses[:, 0] = fitness
            self.fitnesses[:, 1] = func(self.candidates[:, 1])

            # Evolutionary Strategy
            self.candidates[np.random.choice(self.population_size, size=10, replace=False), :] = self.candidates[np.random.choice(self.population_size, size=10, replace=False), :] + \
                                                                                      self.candidates[np.random.choice(self.population_size, size=10, replace=False), :] * \
                                                                                      np.random.uniform(-0.1, 0.1, size=(10, self.dim))

            # Swarm Intelligence
            for _ in range(10):
                new_candidate = np.random.uniform(-5.0, 5.0, self.dim)
                new_fitness = func(new_candidate)
                if new_fitness < self.fitnesses[np.argmin(self.fitnesses[:, 0]), 0] and new_fitness < self.fitnesses[np.argmin(self.fitnesses[:, 1]), 1]:
                    self.best_candidates[np.argmin(self.fitnesses[:, 0]), :] = new_candidate
                    self.fitnesses[np.argmin(self.fitnesses[:, 0]), :] = new_fitness
                    self.fitnesses[np.argmin(self.fitnesses[:, 1]), 1] = new_fitness
                    self.best_candidates[np.argmin(self.fitnesses[:, 1]), :] = new_candidate
                    self.fitnesses[np.argmin(self.fitnesses[:, 1]), 1] = new_fitness

            # Selection
            self.candidates = self.candidates[np.argsort(self.fitnesses[:, 0])]
            self.population_size = self.population_size // 2

            # Mutation
            self.candidates[np.random.choice(self.population_size, size=self.population_size, replace=False), :] += np.random.uniform(-0.1, 0.1, size=(self.population_size, self.dim))

            # Check if the best candidate is improved
            if self.fitnesses[np.argmin(self.fitnesses[:, 0]), 0] < self.fitnesses[np.argmin(self.fitnesses[:, 1]), 1]:
                self.candidates[np.argmin(self.fitnesses[:, 0]), :] = self.best_candidates[np.argmin(self.fitnesses[:, 0]), :]

        return self.best_candidates[np.argmin(self.fitnesses[:, 0]), :], self.fitnesses[np.argmin(self.fitnesses[:, 0]), 0]

# Example usage:
def func(x):
    return x[0]**2 + x[1]**2

multi_objective_ESO = MultiObjectiveEvolutionarySwarmOptimization(budget=100, dim=2)
best_candidate, best_fitness = multi_objective_ESO(func)
print(f"Best candidate: {best_candidate}, Best fitness: {best_fitness}")