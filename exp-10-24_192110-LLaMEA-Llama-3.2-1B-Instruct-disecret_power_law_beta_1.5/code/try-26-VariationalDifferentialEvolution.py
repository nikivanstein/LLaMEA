import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt

class VariationalDifferentialEvolution:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.func = None
        self.population = []
        self.best_func = None
        self.best_score = float('-inf')

    def __call__(self, func):
        # Evaluate the function within the given budget
        func_evals = []
        for _ in range(self.budget):
            func_evals.append(func(torch.randn(self.dim)))

        # Initialize the population with random points in the search space
        self.population = np.random.uniform(-5.0, 5.0, (self.dim, self.budget))

        # Run the differential evolution algorithm
        for _ in range(100):
            # Evaluate the function at each point in the population
            func_evals = [func(x) for x in self.population]

            # Compute the gradients of the function with respect to each point in the population
            gradients = torch.autograd.grad(func_evals, self.population, retain_graph=True)

            # Compute the Varying Differential Evolution update rule
            self.population = self.population + gradients[0].numpy() * 0.01

            # Evaluate the function at each point in the population again
            func_evals = [func(x) for x in self.population]

            # Compute the Varying Differential Evolution update rule again
            self.population = self.population + gradients[0].numpy() * 0.01

            # Check if the current population is better than the best found so far
            if self.best_func is None or np.linalg.norm(self.population - self.best_func) < 1e-6:
                self.best_func = self.population
                self.best_score = np.linalg.norm(self.population - self.best_func)
                print("Best solution found: ", self.best_func)
                print("Best score: ", self.best_score)
                break

        return self.best_func

# Example usage
budget = 1000
dim = 10
best_func = VariationalDifferentialEvolution(budget, dim)
best_func(func)

# Plot the best function and its score
plt.plot(best_func)
plt.show()