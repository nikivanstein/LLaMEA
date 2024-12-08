import numpy as np
import random
from scipy.stats import norm

class LeverageEntropy:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.f_best = None
        self.x_best = None
        self.f_best_val = float('inf')
        self.entropy = 0.0
        self.population_size = 20
        self.population = [self.random_point() for _ in range(self.population_size)]

    def __call__(self, func):
        self.f_best = None
        self.x_best = None
        self.f_best_val = float('inf')
        self.entropy = 0.0

        for _ in range(self.budget):
            # Evaluate the function at each point in the population
            f_values = [func(x) for x in self.population]

            # Update the best solution if the current solution is better
            if self.f_best is None or min(f_values) < self.f_best:
                self.f_best = min(f_values)
                self.x_best = self.population[f_values.index(min(f_values))]
                self.f_best_val = self.f_best

            # Update the entropy
            self.entropy += np.sum(f_values) / self.population_size

            # Replace the worst solution with a new one
            self.population = [self.random_point() if np.random.rand() > 0.5 else x for x in self.population]

        # Reduce the entropy to maintain the balance between exploration and exploitation
        self.entropy = max(0.0, self.entropy - 0.1)

        # Update the best solution if the current solution is better
        if self.f_best_val > self.f_best:
            self.f_best = self.f_best
            self.x_best = self.x_best

        return self.f_best

    def random_point(self):
        return np.random.uniform(self.lower_bound, self.upper_bound, self.dim)

# Example usage
def func(x):
    return np.sum(x ** 2)

budget = 100
dim = 10
leverage_entropy = LeverageEntropy(budget, dim)
for _ in range(100):
    print(leverage_entropy(func))
