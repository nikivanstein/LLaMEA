import numpy as np
import random

class CognitiveSwarmOptimization:
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

            # Cognitive Swarm Intelligence
            for _ in range(10):
                new_candidate = np.random.uniform(-5.0, 5.0, self.dim)
                new_fitness = func(new_candidate)
                if new_fitness < self.best_fitness:
                    self.best_candidate = new_candidate
                    self.best_fitness = new_fitness
                    self.candidates[np.argmin(self.candidates[:, 0]), :] = new_candidate

            # Evolutionary Strategy
            for _ in range(10):
                # Random selection of 10 individuals
                selected_indices = np.random.choice(self.population_size, size=10, replace=False)
                selected_candidates = self.candidates[selected_indices, :]
                # Mutation with probability 0.25
                mutated_candidates = selected_candidates.copy()
                mutated_candidates[np.random.choice(selected_indices, size=int(10*0.25), replace=False), :] += np.random.uniform(-0.1, 0.1, size=(int(10*0.25), self.dim))
                # Selection
                mutated_candidates = mutated_candidates[np.argsort(mutated_candidates[:, 0])]
                # Update the population
                self.candidates[selected_indices, :] = mutated_candidates[:10]

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

cognitive_SCO = CognitiveSwarmOptimization(budget=100, dim=2)
best_candidate, best_fitness = cognitive_SCO(func)
print(f"Best candidate: {best_candidate}, Best fitness: {best_fitness}")