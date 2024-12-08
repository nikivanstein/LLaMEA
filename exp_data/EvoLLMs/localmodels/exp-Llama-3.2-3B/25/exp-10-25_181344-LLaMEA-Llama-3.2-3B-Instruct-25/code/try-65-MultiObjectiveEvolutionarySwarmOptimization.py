import numpy as np
import random
from scipy.optimize import differential_evolution

class MultiObjectiveEvolutionarySwarmOptimization:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 100
        self.candidates = np.random.uniform(-5.0, 5.0, (self.population_size, self.dim))
        self.best_candidate = np.random.uniform(-5.0, 5.0, self.dim)
        self.best_fitness = np.inf

    def __call__(self, func):
        for _ in range(self.budget):
            # Multi-Objective Evolutionary Strategy
            new_candidates = []
            for _ in range(self.population_size):
                candidate = np.random.uniform(-5.0, 5.0, self.dim)
                fitness = func(candidate)
                new_candidates.append((candidate, fitness))
            new_candidates = sorted(new_candidates, key=lambda x: x[1])
            self.candidates = np.array([x[0] for x in new_candidates[:self.population_size//2]])

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

            # Mutation
            self.candidates[np.random.choice(self.population_size, size=self.population_size, replace=False), :] += np.random.uniform(-0.1, 0.1, size=(self.population_size, self.dim))

            # Check if the best candidate is improved
            if self.best_fitness < func(self.best_candidate):
                self.candidates[np.argmin(self.candidates[:, 0]), :] = self.best_candidate

        return self.best_candidate, self.best_fitness

# Example usage:
def func(x):
    return x[0]**2 + x[1]**2

multi_objective_ESO = MultiObjectiveEvolutionarySwarmOptimization(budget=100, dim=2)
best_candidate, best_fitness = multi_objective_ESO(func)
print(f"Best candidate: {best_candidate}, Best fitness: {best_fitness}")