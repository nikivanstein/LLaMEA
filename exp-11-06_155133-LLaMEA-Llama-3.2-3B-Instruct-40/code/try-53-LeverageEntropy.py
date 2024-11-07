import numpy as np
import random

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
        self.population_size = 10

    def __call__(self, func):
        self.f_best = None
        self.x_best = None
        self.f_best_val = float('inf')
        self.entropy = 0.0

        # Initialize the population with random points
        self.population = [np.random.uniform(self.lower_bound, self.upper_bound, self.dim) for _ in range(self.population_size)]

        for _ in range(self.budget):
            # Calculate the entropy of each point in the population
            entropies = [self._calculate_entropy(x) for x in self.population]

            # Select the points with the highest entropy
            self.population = [self.population[np.argsort(entropies)[-2:]]]

            # Evaluate the function at the selected points
            f_values = [func(x) for x in self.population]

            # Update the best solution if the current solution is better
            if self.f_best is None or min(f_values) < self.f_best:
                self.f_best = min(f_values)
                self.x_best = self.population[np.argmin(f_values)]

            # Reduce the entropy to maintain the balance between exploration and exploitation
            self.entropy -= np.mean(entropies) * 0.1

        # Reduce the entropy to maintain the balance between exploration and exploitation
        self.entropy = max(0.0, self.entropy - 0.1)

        # Update the best solution if the current solution is better
        if self.f_best_val > self.f_best:
            self.f_best = self.f_best
            self.x_best = self.x_best

        return self.f_best

    def _calculate_entropy(self, x):
        entropy = 0.0
        for i in range(self.dim):
            if x[i]!= self.lower_bound and x[i]!= self.upper_bound:
                entropy += 1 / np.log(2 * np.pi * np.sqrt(1 + (x[i] - self.lower_bound) ** 2))
        return entropy

# Example usage
def func(x):
    return np.sum(x ** 2)

budget = 100
dim = 10
leverage_entropy = LeverageEntropy(budget, dim)
for _ in range(100):
    print(leverage_entropy(func))
