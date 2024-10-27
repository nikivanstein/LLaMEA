# Description: Efficient Black Box Optimization using Hyperband and Bayesian Optimization
# Code: 
# ```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

class HyperbandBBO:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.func_evals = 0
        self.search_space = (-5.0, 5.0)
        self.search_space_dim = self.dim

    def __call__(self, func, algorithm="Hyperband"):
        if algorithm == "Hyperband":
            # Select the best individual using Hyperband
            if self.budget == 0:
                return np.random.uniform(-5.0, 5.0, self.search_space_dim)
            else:
                new_individual = self.evaluate_fitness(self.func_evals_evals)
                self.func_evals_evals = 0
                for _ in range(self.budget):
                    new_individual = self.evaluate_fitness(new_individual)
                return new_individual
        elif algorithm == "Bayesian":
            # Select the best individual using Bayesian Optimization
            if self.budget == 0:
                return np.random.uniform(-5.0, 5.0, self.search_space_dim)
            else:
                new_individual = self.evaluate_fitness(self.func_evals_evals)
                self.func_evals_evals = 0
                for _ in range(self.budget):
                    new_individual = self.evaluate_fitness(new_individual)
                return new_individual
        else:
            raise ValueError("Invalid algorithm. Please choose 'Hyperband' or 'Bayesian'.")

    def evaluate_fitness(self, func):
        while True:
            # Sample a new point in the search space using Gaussian distribution
            x = np.random.uniform(*self.search_space, size=self.search_space_dim)
            # Evaluate the function at the new point
            func_value = func(x)
            # Store the function value and the new point
            self.func_evals += 1
            self.func_evals_evals = func_value
            # Store the new point in the search space
            self.search_space = (min(self.search_space[0], x), max(self.search_space[1], x))
            # Check if the new point is within the search space
            if x not in self.search_space:
                break

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