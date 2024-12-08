# Description: Evolutionary Optimization using Non-Local Temperature and Adaptive Mutation
# Code: 
import numpy as np
import random
from scipy.optimize import minimize

class NonLocalTemperatureMetaheuristic:
    def __init__(self, budget, dim, alpha=0.5, mu=0.1, tau=0.9):
        self.budget = budget
        self.dim = dim
        self.alpha = alpha
        self.mu = mu
        self.tau = tau
        self.temp = 1.0
        self.best_func = None
        self.x = None

    def __call__(self, func):
        if self.budget <= 0:
            raise ValueError("Budget cannot be zero or negative")

        num_evals = 0
        while num_evals < self.budget and self.best_func is None:
            # Generate a random perturbation
            perturbation = np.random.uniform(-self.dim, self.dim)

            # Evaluate the new function
            new_func = func + perturbation

            # Check if the new function is better
            if np.random.rand() < self.alpha:
                self.best_func = new_func
            else:
                # If the new function is not better, revert the perturbation
                perturbation *= self.tau
                new_func = func + perturbation

            num_evals += 1

        return self.best_func

    def evaluate_fitness(self, func):
        if self.x is None:
            # Initialize the initial solution
            self.x = np.array([0.0] * self.dim)

        # Perform the optimization
        res = minimize(lambda x: func(x), self.x, method="SLSQP", bounds=[(-self.dim, self.dim)], options={"maxiter": 1000})
        self.x = res.x

        # Update the best function
        if np.random.rand() < self.alpha:
            self.best_func = func
        else:
            # If the new function is not better, revert the perturbation
            perturbation = np.random.uniform(-self.dim, self.dim)
            self.x = self.x + perturbation

        return self.best_func

# One-line description: Evolutionary Optimization using Non-Local Temperature and Adaptive Mutation
# Code: 