import random
import numpy as np
import torch

class BlackBoxOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.search_space = (-5.0, 5.0)
        self.func_evaluations = 0
        self.population = None
        self.population_size = 100
        self.population_init = torch.randn(self.population_size, self.dim)
        self.population_optimizer = torch.optim.Adam(self.population_init, lr=0.01)

    def __call__(self, func):
        while self.func_evaluations < self.budget:
            # Generate a random point in the search space
            point = self.population_init.clone()
            # Evaluate the function at the point
            func_value = func(point)
            # Increment the function evaluations
            self.func_evaluations += 1
            # Check if the point is within the budget
            if self.func_evaluations < self.budget:
                # If not, return the point
                return point
            # Refine the strategy by changing the individual lines of the selected solution
            # Update the population using the probability 0.15
            if random.random() < 0.15:
                self.population_init = self.population_init.clone()
                self.population_optimizer = torch.optim.Adam(self.population_init, lr=0.01)
            # Update the population using the probability 0.85
            else:
                self.population_optimizer = torch.optim.Adam(self.population_init, lr=0.001)
            # Evaluate the function at the updated point
            func_value = func(self.population_init)
            # Update the population
            self.population_init = self.population_init + 0.1 * (self.population_init - self.population_init.clone())
            # Check if the updated point is within the budget
            if self.func_evaluations < self.budget:
                # If not, return the updated point
                return self.population_init.clone()
        # If the budget is reached, return the best point found so far
        return self.search_space[0], self.search_space[1]

# Example usage:
if __name__ == "__main__":
    # Initialize the optimizer
    optimizer = BlackBoxOptimizer(budget=100, dim=10)
    # Define the black box function
    def func(x):
        return x**2 + 2*x + 1
    # Optimize the function using the optimizer
    best_point = optimizer(func)
    # Print the best point
    print("Best point:", best_point)