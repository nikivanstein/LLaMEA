# Description: "Black Box Optimization using Genetic Algorithm"
# Code: 
# ```python
import random
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import differential_evolution

class MetaHeuristic:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.search_space = [-5.0, 5.0]  # Define the search space
        self.best_func = None  # Initialize the best function found so far
        self.best_fitness = float('inf')  # Initialize the best fitness found so far
        self.iterations = 0  # Initialize the number of iterations

    def __call__(self, func, max_evals):
        # Evaluate the function up to max_evals times
        for _ in range(max_evals):
            # Randomly sample a point in the search space
            point = (random.uniform(self.search_space[0], self.search_space[1]), random.uniform(self.search_space[0], self.search_space[1]))
            # Evaluate the function at the point
            fitness = func(point)
            # If the fitness is better than the current best fitness, update the best function and fitness
            if fitness < self.best_fitness:
                self.best_func = func
                self.best_fitness = fitness
            # If the fitness is equal to the current best fitness, update the best function if it has a lower budget
            elif fitness == self.best_fitness and self.budget < self.best_func.budget:
                self.best_func = func
                self.best_fitness = fitness
        # Return the best function found
        return self.best_func

    def set_budget(self, budget):
        self.budget = budget

    def get_best_func(self):
        return self.best_func

# Define a noiseless function
def noiseless_func(x):
    return np.sin(x)

# Define a noise function
def noise_func(x):
    return np.random.normal(0, 1, x)

# Define a test function
def test_func(x):
    return x**2 + 2*x + 1

# Create an instance of the MetaHeuristic class
meta_heuristic = MetaHeuristic(100, 10)

# Set the budget for the MetaHeuristic
meta_heuristic.set_budget(100)

# Optimize the test function using the MetaHeuristic
best_func = meta_heuristic(__call__, 100)

# Print the best function found
print("Best function:", best_func)

# Use differential evolution to optimize the test function
def optimize_test_func(x0, bounds, max_evals):
    res = differential_evolution(test_func, [(x - bounds[0], bounds[1]) for x in x0], x0=x0, bounds=bounds, maxiter=max_evals)
    return res.fun

# Print the result of differential evolution
print("Optimized function:", optimize_test_func(np.array([0, 0]), [-5.0, 5.0], 1000))

# Refine the strategy using probability 0.03
def refine_strategy(x0, bounds, max_evals):
    res = differential_evolution(test_func, [(x - bounds[0], bounds[1]) for x in x0], x0=x0, bounds=bounds, maxiter=max_evals)
    return res.fun, res.x

# Print the result of refining the strategy
print("Refined strategy:", refine_strategy(np.array([0, 0]), [-5.0, 5.0], 1000))