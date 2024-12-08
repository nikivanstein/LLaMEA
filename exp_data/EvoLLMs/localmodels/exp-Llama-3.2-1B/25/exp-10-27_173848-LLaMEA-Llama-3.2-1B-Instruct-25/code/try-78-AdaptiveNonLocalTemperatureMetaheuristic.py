import numpy as np
import random

class AdaptiveNonLocalTemperatureMetaheuristic:
    def __init__(self, budget, dim, alpha=0.5, mu=0.1, tau=0.9):
        self.budget = budget
        self.dim = dim
        self.alpha = alpha
        self.mu = mu
        self.tau = tau
        self.temp = 1.0
        self.best_func = None
        self.best_fitness = np.inf

    def __call__(self, func):
        if self.budget <= 0:
            raise ValueError("Budget cannot be zero or negative")

        num_evals = 0
        while num_evals < self.budget and self.best_fitness == np.inf:
            # Generate a random perturbation
            perturbation = np.random.uniform(-self.dim, self.dim)

            # Evaluate the new function
            new_func = func + perturbation

            # Check if the new function is better
            if np.random.rand() < self.alpha:
                self.best_func = new_func
                self.best_fitness = np.min([func(new_func) for new_func in self.evaluate_fitness(self.best_func)])
            else:
                # If the new function is not better, revert the perturbation
                perturbation *= self.tau
                new_func = func + perturbation
                if np.random.rand() < self.alpha:
                    self.best_func = new_func
                    self.best_fitness = np.min([func(new_func) for new_func in self.evaluate_fitness(self.best_func)])

            num_evals += 1

        return self.best_func

    def evaluate_fitness(self, func):
        # Evaluate the function at a specified number of points
        num_points = min(10, self.budget // 0.25)  # 10 points with 0.25 probability
        points = np.random.choice(self.dim, num_points, replace=False)
        return func(points)

# One-line description: Evolutionary Optimization using Adaptive Non-Local Temperature and Adaptive Mutation
# Code: 