import numpy as np
import random

class MultiObjectiveEvolutionaryOptimizationWithDifferentialEvolution:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 100
        self.candidates = np.random.uniform(-5.0, 5.0, (self.population_size, self.dim))
        self.best_candidate = np.random.uniform(-5.0, 5.0, self.dim)
        self.best_fitness = np.inf
        self.mutation_probability = 0.25

    def __call__(self, func):
        for _ in range(self.budget):
            fitness = func(self.candidates[:, 0])
            self.best_candidate = self.candidates[np.argmin(fitness), :]
            self.best_fitness = np.min(fitness)

            # Multi-Objective Optimization
            for _ in range(10):
                new_candidate = np.random.uniform(-5.0, 5.0, self.dim)
                new_fitness = func(new_candidate)
                if new_fitness < self.best_fitness:
                    self.best_candidate = new_candidate
                    self.best_fitness = new_fitness
                    self.candidates[np.argmin(fitness), :] = new_candidate

            # Differential Evolution
            for i in range(self.population_size):
                r1 = np.random.randint(0, self.population_size)
                r2 = np.random.randint(0, self.population_size)
                r3 = np.random.randint(0, self.population_size)

                x1, x2, x3 = self.candidates[i], self.candidates[r1], self.candidates[r2]
                x1, x2, x3 = x1 + np.random.uniform(-0.1, 0.1, size=self.dim), x2 + np.random.uniform(-0.1, 0.1, size=self.dim), x3 + np.random.uniform(-0.1, 0.1, size=self.dim)

                f1, f2, f3 = func(x1), func(x2), func(x3)
                f1, f2, f3 = f1 + np.random.uniform(-0.1, 0.1, size=self.dim), f2 + np.random.uniform(-0.1, 0.1, size=self.dim), f3 + np.random.uniform(-0.1, 0.1, size=self.dim)

                x1, x2, x3 = x1 - f1 + f2, x2 - f2 + f3, x3 - f3 + f1
                x1, x2, x3 = x1 / (np.linalg.norm(x1) + np.linalg.norm(x2) + np.linalg.norm(x3)), x2 / (np.linalg.norm(x1) + np.linalg.norm(x2) + np.linalg.norm(x3)), x3 / (np.linalg.norm(x1) + np.linalg.norm(x2) + np.linalg.norm(x3))

                if np.random.rand() < self.mutation_probability:
                    self.candidates[i] = x1
                    self.candidates[r1] = x2
                    self.candidates[r2] = x3

            # Selection
            self.candidates = self.candidates[np.argsort(fitness)]
            self.population_size = self.population_size // 2

            # Mutation
            self.candidates[np.random.choice(self.population_size, size=self.population_size, replace=False), :] += np.random.uniform(-0.1, 0.1, size=(self.population_size, self.dim))

            # Check if the best candidate is improved
            if self.best_fitness < func(self.best_candidate):
                self.candidates[np.argmin(fitness), :] = self.best_candidate

        return self.best_candidate, self.best_fitness

# Example usage:
def func(x):
    return x[0]**2 + x[1]**2

multi_objective_DE = MultiObjectiveEvolutionaryOptimizationWithDifferentialEvolution(budget=100, dim=2)
best_candidate, best_fitness = multi_objective_DE(func)
print(f"Best candidate: {best_candidate}, Best fitness: {best_fitness}")