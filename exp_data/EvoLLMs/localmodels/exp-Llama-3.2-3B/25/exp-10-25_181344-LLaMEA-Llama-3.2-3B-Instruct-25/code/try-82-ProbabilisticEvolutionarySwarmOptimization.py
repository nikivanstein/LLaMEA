import numpy as np
import random

class ProbabilisticEvolutionarySwarmOptimization:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 100
        self.candidates = np.random.uniform(-5.0, 5.0, (self.population_size, self.dim))
        self.best_candidate = np.random.uniform(-5.0, 5.0, self.dim)
        self.best_fitness = np.inf
        self.probability = 0.25

    def __call__(self, func):
        for _ in range(self.budget):
            fitness = func(self.candidates[:, 0])
            self.best_candidate = self.candidates[np.argmin(self.candidates[:, 0]), :]
            self.best_fitness = fitness

            # Evolutionary Strategy
            new_candidates = []
            for _ in range(10):
                new_candidate = self.candidates[np.random.choice(self.population_size, size=1, replace=False), :]
                mutation = np.random.uniform(-0.1, 0.1, size=(1, self.dim))
                new_candidate += mutation
                new_candidates.append(new_candidate)
            new_candidates = np.array(new_candidates)
            self.candidates[np.random.choice(self.population_size, size=len(new_candidates), replace=False), :] = new_candidates

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

            # Refine strategy with probability
            if np.random.rand() < self.probability:
                for i in range(self.population_size):
                    mutation = np.random.uniform(-0.1, 0.1, size=(1, self.dim))
                    self.candidates[i, :] += mutation

            # Check if the best candidate is improved
            if self.best_fitness < func(self.best_candidate):
                self.candidates[np.argmin(self.candidates[:, 0]), :] = self.best_candidate

        return self.best_candidate, self.best_fitness

# Example usage:
def func(x):
    return x[0]**2 + x[1]**2

probabilistic_ESO = ProbabilisticEvolutionarySwarmOptimization(budget=100, dim=2)
best_candidate, best_fitness = probabilistic_ESO(func)
print(f"Best candidate: {best_candidate}, Best fitness: {best_fitness}")