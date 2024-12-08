# Description: Novel Metaheuristic for Black Box Optimization
# Code: 
# ```python
import random
import numpy as np
from scipy.optimize import minimize
from typing import Dict, List

class MetaHeuristic:
    def __init__(self, budget: int, dim: int):
        self.budget = budget
        self.dim = dim
        self.search_space = [-5.0, 5.0]  # Define the search space
        self.best_func = None  # Initialize the best function found so far
        self.best_fitness = float('inf')  # Initialize the best fitness found so far
        self.iterations = 0  # Initialize the number of iterations

    def __call__(self, func: callable, max_evals: int) -> Dict:
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
        return {"best_func": self.best_func, "best_fitness": self.best_fitness}

    def set_budget(self, budget: int):
        self.budget = budget

    def get_best_func(self):
        return self.best_func

# Define a noiseless function
def noiseless_func(x: np.ndarray) -> float:
    return np.sin(x)

# Define a noise function
def noise_func(x: np.ndarray) -> float:
    return np.random.normal(0, 1, x)

# Define a test function
def test_func(x: np.ndarray) -> float:
    return x**2 + 2*x + 1

# Create an instance of the MetaHeuristic class
meta_heuristic = MetaHeuristic(100, 10)

# Set the budget for the MetaHeuristic
meta_heuristic.set_budget(100)

# Optimize the test function using the MetaHeuristic
best_func = meta_heuristic(__call__, 100)

# Print the best function found
print("Best function:", best_func)

# Refine the strategy by changing the individual lines
def optimize_bbob(x: np.ndarray, max_evals: int, budget: int) -> Dict:
    # Evaluate the function up to max_evals times
    for _ in range(max_evals):
        # Randomly sample a point in the search space
        point = (x + random.uniform(-5.0, 5.0), x + random.uniform(-5.0, 5.0))
        # Evaluate the function at the point
        fitness = noiseless_func(point)
        # If the fitness is better than the current best fitness, update the best function and fitness
        if fitness < best_func["best_fitness"]:
            best_func = {"best_func": best_func["best_func"], "best_fitness": fitness}
        # If the fitness is equal to the current best fitness, update the best function if it has a lower budget
        elif fitness == best_func["best_fitness"] and best_func["budget"] < budget:
            best_func = {"best_func": best_func["best_func"], "best_fitness": fitness}
        # Update the individual lines
        x = (x + random.uniform(-5.0, 5.0), x + random.uniform(-5.0, 5.0))
    # Return the best function found
    return best_func

# Refine the optimization using the new individual lines
best_func = optimize_bbob(test_func(np.array([0])), 100, 100)

# Print the best function found
print("Best function:", best_func)