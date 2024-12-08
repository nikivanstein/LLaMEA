import numpy as np
import random

class HyperNetworkEvolutionaryOptimization:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 100
        self.candidates = np.random.uniform(-5.0, 5.0, (self.population_size, self.dim))
        self.best_candidate = np.random.uniform(-5.0, 5.0, self.dim)
        self.best_fitness = np.inf
        self.hyper_network = np.random.uniform(-5.0, 5.0, (self.population_size, self.dim))

    def __call__(self, func):
        for _ in range(self.budget):
            fitness = func(self.candidates[:, 0])
            self.best_candidate = self.candidates[np.argmin(self.candidates[:, 0]), :]
            self.best_fitness = fitness

            # Hyperparameter Tuning
            self.hyper_network[np.random.choice(self.population_size, size=10, replace=False), :] = self.hyper_network[np.random.choice(self.population_size, size=10, replace=False), :] + \
                                                                                      self.hyper_network[np.random.choice(self.population_size, size=10, replace=False), :] * \
                                                                                      np.random.uniform(-0.1, 0.1, size=(10, self.dim))

            # Evolutionary Strategy
            self.candidates[np.random.choice(self.population_size, size=10, replace=False), :] = self.candidates[np.random.choice(self.population_size, size=10, replace=False), :] + \
                                                                                      self.candidates[np.random.choice(self.population_size, size=10, replace=False), :] * \
                                                                                      np.random.uniform(-0.1, 0.1, size=(10, self.dim))

            # Selection
            self.candidates = self.candidates[np.argsort(self.candidates[:, 0])]
            self.population_size = self.population_size // 2

            # Mutation
            self.candidates[np.random.choice(self.population_size, size=self.population_size, replace=False), :] += np.random.uniform(-0.1, 0.1, size=(self.population_size, self.dim))

            # Hyperparameter Refinement
            for _ in range(10):
                new_hyper_network = np.random.uniform(-5.0, 5.0, (self.population_size, self.dim))
                new_fitness = func(new_hyper_network[:, 0])
                if new_fitness < self.best_fitness:
                    self.best_candidate = new_hyper_network[np.argmin(new_hyper_network[:, 0]), :]
                    self.best_fitness = new_fitness
                    self.candidates[np.argmin(self.candidates[:, 0]), :] = new_hyper_network[np.argmin(new_hyper_network[:, 0]), :]

            # Check if the best candidate is improved
            if self.best_fitness < func(self.best_candidate):
                self.candidates[np.argmin(self.candidates[:, 0]), :] = self.best_candidate

        return self.best_candidate, self.best_fitness

# Example usage:
def func(x):
    return x[0]**2 + x[1]**2

hyperNEO = HyperNetworkEvolutionaryOptimization(budget=100, dim=2)
best_candidate, best_fitness = hyperNEO(func)
print(f"Best candidate: {best_candidate}, Best fitness: {best_fitness}")