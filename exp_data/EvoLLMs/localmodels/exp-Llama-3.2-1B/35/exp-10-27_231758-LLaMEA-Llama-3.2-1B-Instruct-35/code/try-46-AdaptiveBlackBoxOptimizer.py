import numpy as np
import random

class AdaptiveBlackBoxOptimizer:
    def __init__(self, budget, dim, mutation_rate=0.01, exploration_rate=0.5):
        self.budget = budget
        self.dim = dim
        self.func_evals = 0
        self.func_values = None
        self.mutation_rate = mutation_rate
        self.exploration_rate = exploration_rate
        self.population_size = 100
        self.population = np.random.rand(self.population_size, self.dim)

    def __call__(self, func):
        if self.func_values is None:
            self.func_evals = self.budget
            self.func_values = np.zeros(self.dim)
            for _ in range(self.func_evals):
                func(self.func_values)
        else:
            while self.func_evals > 0:
                idx = np.argmin(np.abs(self.func_values))
                self.func_values[idx] = func(self.func_values[idx])
                self.func_evals -= 1
                if self.func_evals == 0:
                    break

        # Refine the solution
        if random.random() < self.exploration_rate:
            idx = random.randint(0, self.dim - 1)
            self.func_values[idx] = func(self.func_values[idx])
        if random.random() < self.mutation_rate:
            idx = random.randint(0, self.dim - 1)
            self.func_values[idx] = func(self.func_values[idx] + random.uniform(-1, 1))

        # Evaluate the new solution
        func_values = np.array([func(self.func_values[i]) for i in range(self.dim)])
        self.func_values = func_values
        self.func_evals += 1

        # Update the population
        if self.func_evals >= self.budget:
            self.population = np.random.rand(self.population_size, self.dim)
            self.func_evals = 0