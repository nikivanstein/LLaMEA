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
        self.population = [np.random.uniform(self.lower_bound, self.upper_bound, self.dim) for _ in range(self.population_size)]

        for _ in range(self.budget):
            # Select the best solution from the population
            f_values = [func(x) for x in self.population]
            self.f_best = min(f_values)
            self.x_best = self.population[f_values.index(self.f_best)]

            # Generate a new solution by leveraging the best solution
            x = self.x_best.copy()
            for i in range(self.dim):
                if np.random.rand() < 0.5:
                    x[i] += np.random.uniform(-1.0, 1.0)
                    x[i] = max(self.lower_bound, min(x[i], self.upper_bound))

            # Add the new solution to the population
            self.population.append(x)

            # Calculate the entropy of the population
            entropy = 0.0
            for x in self.population:
                for i in range(self.dim):
                    if x[i]!= self.lower_bound and x[i]!= self.upper_bound:
                        entropy += 1 / np.log(2 * np.pi * np.sqrt(1 + (x[i] - self.lower_bound) ** 2))
            self.entropy += entropy / self.population_size

            # Reduce the entropy to maintain the balance between exploration and exploitation
            self.entropy = max(0.0, self.entropy - 0.1)

        # Update the best solution if the current solution is better
        if self.f_best_val > self.f_best:
            self.f_best = self.f_best
            self.x_best = self.x_best

        return self.f_best

# Example usage
def func(x):
    return np.sum(x ** 2)

budget = 100
dim = 10
leverage_entropy = LeverageEntropy(budget, dim)
for _ in range(100):
    print(leverage_entropy(func))
