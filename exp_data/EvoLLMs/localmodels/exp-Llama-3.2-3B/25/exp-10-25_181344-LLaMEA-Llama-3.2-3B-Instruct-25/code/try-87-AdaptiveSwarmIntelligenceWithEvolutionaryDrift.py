import numpy as np
import random

class AdaptiveSwarmIntelligenceWithEvolutionaryDrift:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 100
        self.candidates = np.random.uniform(-5.0, 5.0, (self.population_size, self.dim))
        self.best_candidate = np.random.uniform(-5.0, 5.0, self.dim)
        self.best_fitness = np.inf
        self.swarm_intelligence_coefficient = 0.25
        self.evolutionary_drift_coefficient = 0.1

    def __call__(self, func):
        for _ in range(self.budget):
            fitness = func(self.candidates[:, 0])
            self.best_candidate = self.candidates[np.argmin(self.candidates[:, 0]), :]
            self.best_fitness = fitness

            # Adaptive Swarm Intelligence
            for _ in range(10):
                new_candidate = self.candidates[np.random.choice(self.population_size, size=1, replace=False), :]
                new_fitness = func(new_candidate)
                if new_fitness < self.best_fitness * (1 + np.random.uniform(-self.swarm_intelligence_coefficient, self.swarm_intelligence_coefficient)):
                    self.best_candidate = new_candidate
                    self.best_fitness = new_fitness
                    self.candidates[np.argmin(self.candidates[:, 0]), :] = new_candidate

            # Evolutionary Drift
            for _ in range(10):
                new_candidate = self.candidates[np.random.choice(self.population_size, size=1, replace=False), :]
                new_fitness = func(new_candidate)
                if new_fitness < self.best_fitness * (1 + np.random.uniform(-self.evolutionary_drift_coefficient, self.evolutionary_drift_coefficient)):
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

        return self.best_candidate, self.best_fitness

# Example usage:
def func(x):
    return x[0]**2 + x[1]**2

adaptive_SiED = AdaptiveSwarmIntelligenceWithEvolutionaryDrift(budget=100, dim=2)
best_candidate, best_fitness = adaptive_SiED(func)
print(f"Best candidate: {best_candidate}, Best fitness: {best_fitness}")