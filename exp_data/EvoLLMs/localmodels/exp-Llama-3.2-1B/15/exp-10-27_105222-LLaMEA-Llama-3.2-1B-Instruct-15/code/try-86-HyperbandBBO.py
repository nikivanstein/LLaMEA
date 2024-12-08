import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.stats import norm

class HyperbandBBO:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.func_evals = 0
        self.search_space = (-5.0, 5.0)
        self.search_space_dim = self.dim

    def __call__(self, func):
        while self.func_evals < self.budget:
            # Sample a new point in the search space using Gaussian distribution
            x = np.random.uniform(*self.search_space, size=self.search_space_dim)
            # Evaluate the function at the new point
            func_value = func(x)
            # Store the function value and the new point
            self.func_evals += 1
            self.func_evals_evals = func_value
            # Store the new point in the search space
            self.search_space = (min(self.search_space[0], x), max(self.search_space[1], x))
        # Evaluate the function at the final point in the search space
        func_value = func(self.search_space)
        return func_value

    def __str__(self):
        return f"HyperbandBBO(budget={self.budget}, dim={self.dim}, search_space={self.search_space})"

# Example usage:
def test_func1(x):
    return np.sin(x)

def test_func2(x):
    return x**2 + 2*x + 1

hyperband = HyperbandBBO(budget=100, dim=10)
optimized_func1 = hyperband(test_func1)
optimized_func2 = hyperband(test_func2)

# Plot the results
plt.figure(figsize=(8, 6))
plt.plot([optimized_func1, optimized_func2], label=['Test Function 1', 'Test Function 2'])
plt.xlabel('Optimized Function Value')
plt.ylabel('Dimensionality')
plt.title('Black Box Optimization using Hyperband and Bayesian Optimization')
plt.legend()
plt.show()

# Refine the strategy
def refine_strategy(individual):
    # Calculate the fitness of the individual
    fitness = hyperband(test_func1, individual)
    # Calculate the new individual using Bayesian optimization
    new_individual = individual
    while True:
        # Sample a new point in the search space using Gaussian distribution
        x = np.random.uniform(*new_individual.search_space, size=new_individual.search_space_dim)
        # Evaluate the function at the new point
        func_value = test_func1(x)
        # Update the new individual using Bayesian optimization
        new_individual = individual, func_value
        # Store the new point in the search space
        new_individual.search_space = (min(new_individual.search_space[0], x), max(new_individual.search_space[1], x))
        # Evaluate the function at the final point in the search space
        func_value = test_func1(new_individual.search_space)
        # Check if the new fitness is better
        if func_value > fitness:
            return new_individual

optimized_func1 = refine_strategy(optimized_func1)
optimized_func2 = refine_strategy(optimized_func2)

# Plot the results
plt.figure(figsize=(8, 6))
plt.plot([optimized_func1, optimized_func2], label=['Test Function 1', 'Test Function 2'])
plt.xlabel('Optimized Function Value')
plt.ylabel('Dimensionality')
plt.title('Black Box Optimization using Hyperband and Bayesian Optimization')
plt.legend()
plt.show()