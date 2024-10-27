import numpy as np
import random

class PSO_EvolutionaryStrategy:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 100
        self.candidates = np.random.uniform(-5.0, 5.0, (self.population_size, self.dim))
        self.best_candidate = np.random.uniform(-5.0, 5.0, self.dim)
        self.best_fitness = np.inf
        self.pbest = np.zeros((self.population_size, self.dim))
        self.rbest = np.zeros((self.population_size, self.dim))
        self.swarm_intelligence = np.zeros((self.population_size, self.dim))

    def __call__(self, func):
        for _ in range(self.budget):
            fitness = func(self.candidates[:, 0])
            self.best_candidate = self.candidates[np.argmin(self.candidates[:, 0]), :]
            self.best_fitness = fitness

            # Particle Swarm Optimization
            for i in range(self.population_size):
                r1 = np.random.uniform(0, 1)
                r2 = np.random.uniform(0, 1)
                if r1 < 0.25:
                    self.candidates[i, :] += self.swarm_intelligence[i, :] * np.random.uniform(-0.1, 0.1, size=self.dim)
                if r2 < 0.25:
                    self.candidates[i, :] += self.pbest[i, :] * np.random.uniform(-0.1, 0.1, size=self.dim)
                self.swarm_intelligence[i, :] = (self.swarm_intelligence[i, :] * 0.9 + self.candidates[i, :] * 0.1)
                self.pbest[i, :] = self.candidates[i, :]
                self.rbest[i, :] = self.candidates[i, :]

                # Evolutionary Strategy
                self.candidates[i, :] += self.candidates[i, :] * np.random.uniform(-0.1, 0.1, size=self.dim)

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

pso_es = PSO_EvolutionaryStrategy(budget=100, dim=2)
best_candidate, best_fitness = pso_es(func)
print(f"Best candidate: {best_candidate}, Best fitness: {best_fitness}")