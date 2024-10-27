import numpy as np
import random
import copy

class MetaLEHybridEvolutionaryOptimization:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 100
        self.candidates = np.random.uniform(-5.0, 5.0, (self.population_size, self.dim))
        self.best_candidate = np.random.uniform(-5.0, 5.0, self.dim)
        self.best_fitness = np.inf
        self.meta_learning = True
        self.meta_model = None

    def __call__(self, func):
        for _ in range(self.budget):
            if self.meta_learning:
                # Meta-learning step
                self.meta_model = self._meta_learn(func, self.candidates)
                self.meta_learning = False

            fitness = func(self.candidates[:, 0])
            self.best_candidate = self.candidates[np.argmin(self.candidates[:, 0]), :]
            self.best_fitness = fitness

            # Evolutionary Strategy
            self.candidates[np.random.choice(self.population_size, size=10, replace=False), :] = self.candidates[np.random.choice(self.population_size, size=10, replace=False), :] + \
                                                                                      self.candidates[np.random.choice(self.population_size, size=10, replace=False), :] * \
                                                                                      np.random.uniform(-0.1, 0.1, size=(10, self.dim))

            # Swarm Intelligence
            for _ in range(10):
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

            # Check if the best candidate is improved
            if self.best_fitness < func(self.best_candidate):
                self.candidates[np.argmin(self.candidates[:, 0]), :] = self.best_candidate

            if self.meta_learning:
                # Update meta-model
                self.meta_model = self._meta_update(func, self.meta_model, self.candidates)

    def _meta_learn(self, func, candidates):
        # Simple meta-learning model: average fitness of candidates
        return np.mean(func(candidates[:, 0]))

    def _meta_update(self, func, meta_model, candidates):
        # Update meta-model by averaging fitness of new candidates
        new_candidates = np.random.uniform(-5.0, 5.0, (100, self.dim))
        new_fitness = func(new_candidates)
        return np.mean(new_fitness)

# Example usage:
def func(x):
    return x[0]**2 + x[1]**2

meta_LE_HESO = MetaLEHybridEvolutionaryOptimization(budget=100, dim=2)
best_candidate, best_fitness = meta_LE_HESO(func)
print(f"Best candidate: {best_candidate}, Best fitness: {best_fitness}")