import numpy as np
import random
from scipy.optimize import minimize

class QuantumInspiredHybridEvolutionarySwarmOptimization:
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
            self.candidates[np.random.choice(self.population_size, size=10, replace=True), :] = self.candidates[np.random.choice(self.population_size, size=10, replace=True), :] + \
                                                                                      self.candidates[np.random.choice(self.population_size, size=10, replace=True), :] * \
                                                                                      np.random.uniform(-0.1, 0.1, size=(10, self.dim))

            # Quantum Computing
            quantum_candidates = self.candidates.copy()
            for i in range(self.population_size):
                quantum_candidates[i] = quantum_candidates[i] + np.random.normal(0, 0.1, size=self.dim)
                quantum_candidates[i] = np.clip(quantum_candidates[i], -5.0, 5.0)
            quantum_fitness = func(quantum_candidates[:, 0])
            self.candidates[np.argmin(quantum_fitness), :] = quantum_candidates[np.argmin(quantum_fitness), :]

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
            self.candidates[np.random.choice(self.population_size, size=self.population_size, replace=True), :] += np.random.uniform(-0.1, 0.1, size=(self.population_size, self.dim))

            # Check if the best candidate is improved
            if self.best_fitness < func(self.best_candidate):
                self.candidates[np.argmin(self.candidates[:, 0]), :] = self.best_candidate

        return self.best_candidate, self.best_fitness

# Example usage:
def func(x):
    return x[0]**2 + x[1]**2

hybrid_ESO = QuantumInspiredHybridEvolutionarySwarmOptimization(budget=100, dim=2)
best_candidate, best_fitness = hybrid_ESO(func)
print(f"Best candidate: {best_candidate}, Best fitness: {best_fitness}")