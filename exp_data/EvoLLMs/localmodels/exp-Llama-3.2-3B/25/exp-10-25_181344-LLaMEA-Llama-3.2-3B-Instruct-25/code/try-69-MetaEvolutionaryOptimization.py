import numpy as np
import random
import copy

class MetaEvolutionaryOptimization:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 100
        self.candidates = np.random.uniform(-5.0, 5.0, (self.population_size, self.dim))
        self.best_candidate = np.random.uniform(-5.0, 5.0, self.dim)
        self.best_fitness = np.inf
        self.metalog = []

    def __call__(self, func):
        for _ in range(self.budget):
            fitness = func(self.candidates[:, 0])
            self.best_candidate = self.candidates[np.argmin(self.candidates[:, 0]), :]
            self.best_fitness = fitness

            # Evolutionary Strategy
            self.candidates[np.random.choice(self.population_size, size=10, replace=True), :] = self.candidates[np.random.choice(self.population_size, size=10, replace=True), :] + \
                                                                                      self.candidates[np.random.choice(self.population_size, size=10, replace=True), :] * \
                                                                                      np.random.uniform(-0.1, 0.1, size=(10, self.dim))

            # Meta-Learning
            new_metalog = []
            for i in range(self.population_size):
                new_individual = copy.deepcopy(self.candidates[i])
                for j in range(self.dim):
                    if random.random() < 0.25:
                        new_individual[j] += np.random.uniform(-0.1, 0.1)
                new_fitness = func(new_individual)
                if new_fitness < self.best_fitness:
                    self.best_candidate = new_individual
                    self.best_fitness = new_fitness
                new_metalog.append((new_individual, new_fitness))
            self.metalog.append(new_metalog)

            # Selection
            self.candidates = self.candidates[np.argsort(self.candidates[:, 0])]
            self.population_size = self.population_size // 2

            # Mutation
            self.candidates[np.random.choice(self.population_size, size=self.population_size, replace=True), :] += np.random.uniform(-0.1, 0.1, size=(self.population_size, self.dim))

            # Check if the best candidate is improved
            if self.best_fitness < func(self.best_candidate):
                self.candidates[np.argmin(self.candidates[:, 0]), :] = self.best_candidate

        return self.best_candidate, self.best_fitness

# Example usage:
def func(x):
    return x[0]**2 + x[1]**2

meta_EO = MetaEvolutionaryOptimization(budget=100, dim=2)
best_candidate, best_fitness = meta_EO(func)
print(f"Best candidate: {best_candidate}, Best fitness: {best_fitness}")