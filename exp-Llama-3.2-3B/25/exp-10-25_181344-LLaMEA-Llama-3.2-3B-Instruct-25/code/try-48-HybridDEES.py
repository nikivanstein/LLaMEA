import numpy as np
import random

class HybridDEES:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 100
        self.candidates = np.random.uniform(-5.0, 5.0, (self.population_size, self.dim))
        self.best_candidate = np.random.uniform(-5.0, 5.0, self.dim)
        self.best_fitness = np.inf
        self.p = 0.25  # probability of changing individual lines

    def __call__(self, func):
        for _ in range(self.budget):
            # Selection
            self.candidates = self.candidates[np.argsort(func(self.candidates[:, 0]))]

            # Hybrid Evolutionary Strategies
            for i in range(self.population_size):
                if random.random() < self.p:
                    # Change individual lines
                    self.candidates[i, :] += np.random.uniform(-0.1, 0.1, size=self.dim)
                    self.candidates[i, :] = np.clip(self.candidates[i, :], -5.0, 5.0)

            # Differential Evolution
            for i in range(self.population_size):
                # Generate a random vector
                F = np.random.uniform(0.0, 2.0, size=self.dim)

                # Generate a random individual
                r = np.random.uniform(0.0, 1.0, size=self.dim)
                x1 = self.candidates[i, :]
                x2 = self.candidates[(i + 1) % self.population_size, :]
                x3 = self.candidates[(i + 2) % self.population_size, :]
                x4 = self.candidates[(i + 3) % self.population_size, :]

                # Calculate the new individual
                new_x = x1 + F * (x2 - x1) + r * (x3 - x1) + (1 - r) * (x4 - x1)

                # Calculate the fitness
                new_fitness = func(new_x)

                # Update the best candidate
                if new_fitness < self.best_fitness:
                    self.best_candidate = new_x
                    self.best_fitness = new_fitness

            # Update the best candidate
            if self.best_fitness < func(self.best_candidate):
                self.candidates[np.argmin(func(self.candidates[:, 0])), :] = self.best_candidate

        return self.best_candidate, self.best_fitness

# Example usage:
def func(x):
    return x[0]**2 + x[1]**2

hybrid_DEES = HybridDEES(budget=100, dim=2)
best_candidate, best_fitness = hybrid_DEES(func)
print(f"Best candidate: {best_candidate}, Best fitness: {best_fitness}")