import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from hyperband import Hyperband

class EfficientBBO:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.func_evals = 0
        self.search_space = (-5.0, 5.0)
        self.search_space_dim = self.dim
        self.algorithms = {}

    def __call__(self, func):
        # Select the most suitable algorithm based on the budget and dimensionality
        algorithm = self.select_algorithm(func, self.budget, self.dim)
        # Run the selected algorithm
        new_individual = algorithm.evaluate_fitness(func, self)
        # Evaluate the function at the final point in the search space
        func_value = func(new_individual)
        return func_value

    def select_algorithm(self, func, budget, dim):
        # Hyperband
        if np.random.rand() < 0.5:
            algorithm = Hyperband(budget, dim)
        # Bayesian Optimization
        else:
            algorithm = BayesianOptimization(budget, dim)
        # Store the selected algorithm
        self.algorithms[func] = algorithm
        return algorithm

    def bayesian_optimization(self, budget, dim):
        # Initialize the algorithm
        algorithm = BayesianOptimization(budget, dim)
        # Run the algorithm
        new_individual = algorithm.optimize(func, x0=np.random.uniform(*self.search_space, size=self.search_space_dim))
        # Evaluate the function at the final point in the search space
        func_value = func(new_individual)
        return func_value

    def hyperband(self, budget, dim):
        # Initialize the algorithm
        algorithm = Hyperband(budget, dim)
        # Run the algorithm
        new_individual = algorithm.optimize(func, x0=np.random.uniform(*self.search_space, size=self.search_space_dim))
        # Evaluate the function at the final point in the search space
        func_value = func(new_individual)
        return func_value

# Example usage:
def test_func1(x):
    return np.sin(x)

def test_func2(x):
    return x**2 + 2*x + 1

efficient_bbo = EfficientBBO(budget=100, dim=10)
optimized_func1 = efficient_bbo(test_func1)
optimized_func2 = efficient_bbo(test_func2)

# Plot the results
plt.figure(figsize=(8, 6))
plt.plot([optimized_func1, optimized_func2], label=['Test Function 1', 'Test Function 2'])
plt.xlabel('Optimized Function Value')
plt.ylabel('Dimensionality')
plt.title('Black Box Optimization using EfficientBBO')
plt.legend()
plt.show()