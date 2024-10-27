import numpy as np
import random

class ProbabilityBasedEvolutionarySwarmOptimization:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 100
        self.candidates = np.random.uniform(-5.0, 5.0, (self.population_size, self.dim))
        self.best_candidate = np.random.uniform(-5.0, 5.0, self.dim)
        self.best_fitness = np.inf

    def __call__(self, func):
        for _ in range(self.budget):
            fitness = func(self.candidates[:, 0])
            self.best_candidate = self.candidates[np.argmin(self.candidates[:, 0]), :]
            self.best_fitness = fitness

            # Evolutionary Strategy
            new_candidates = []
            for _ in range(10):
                individual = self.candidates[np.random.choice(self.population_size, size=1, replace=False), :]
                mutation = np.random.uniform(-0.1, 0.1, size=(1, self.dim))
                new_individual = individual + mutation
                new_candidates.append(new_individual)

            # Swarm Intelligence
            new_candidates = np.array(new_candidates)
            new_fitness = func(new_candidates)
            new_indices = np.argsort(new_fitness)
            for i in new_indices[:10]:
                if new_fitness[i] < self.best_fitness:
                    self.best_candidate = new_candidates[i]
                    self.best_fitness = new_fitness[i]
                    self.candidates[np.argmin(self.candidates[:, 0]), :] = new_candidates[i]

            # Selection
            new_candidates = new_candidates[np.argsort(new_fitness)]
            self.candidates = new_candidates[:self.population_size//2]

            # Mutation
            self.candidates += np.random.uniform(-0.1, 0.1, size=(self.population_size, self.dim))

            # Check if the best candidate is improved
            if self.best_fitness < func(self.best_candidate):
                self.candidates[np.argmin(self.candidates[:, 0]), :] = self.best_candidate

        return self.best_candidate, self.best_fitness

# Example usage:
def func(x):
    return x[0]**2 + x[1]**2

pbeso = ProbabilityBasedEvolutionarySwarmOptimization(budget=100, dim=2)
best_candidate, best_fitness = pbeso(func)
print(f"Best candidate: {best_candidate}, Best fitness: {best_fitness}")