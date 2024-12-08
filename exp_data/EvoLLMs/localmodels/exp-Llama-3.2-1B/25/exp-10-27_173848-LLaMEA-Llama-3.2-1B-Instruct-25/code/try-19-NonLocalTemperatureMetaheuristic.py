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
        self.population = []

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

            # Add the new individual to the population
            self.population.append(new_func)

            num_evals += 1

        # Evaluate the population to get the best individual
        self.best_func = min(self.population, key=func)

        return self.best_func

# One-line description: Evolutionary Optimization using Non-Local Temperature and Adaptive Mutation
# Code: 
# ```python
# NonLocalTemperatureMetaheuristic: Evolutionary Optimization using Non-Local Temperature and Adaptive Mutation
# ```
# ```python
def optimize_bbob(func, budget, dim, alpha=0.5, mu=0.1, tau=0.9):
    """Optimize a black box function using Non-Local Temperature and Adaptive Mutation"""
    metaheuristic = NonLocalTemperatureMetaheuristic(budget, dim, alpha, mu, tau)
    best_func = metaheuristic(func)
    return best_func

# Evaluate the BBOB test suite of 24 noiseless functions
# ```python
# BBOB test suite of 24 noiseless functions
# ```
# ```python
# import numpy as np
# import random
# from scipy.optimize import minimize

# def func(x):
#     return x[0]**2 + x[1]**2 + x[2]**2 + x[3]**2 + x[4]**2 + x[5]**2

# best_func = optimize_bbob(func, 1000, 6)
# print(best_func)