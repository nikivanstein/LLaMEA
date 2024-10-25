# Description: Novel Metaheuristic Algorithm for Black Box Optimization
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
print("Best fitness:", best_func.budget)

# Define a noiseless function with a different search space
def noiseless_func2(x):
    return np.sin(x / 2) + 0.5 * np.sin(2 * np.pi * x)

# Define a noise function with a different search space
def noise_func2(x):
    return np.random.normal(0, 1, x)

# Define a test function with a different search space
def test_func2(x):
    return x**2 + 2*x + 1 + 0.5 * np.sin(2 * np.pi * x)

# Create an instance of the MetaHeuristic class
meta_heuristic2 = MetaHeuristic(100, 10)

# Set the budget for the MetaHeuristic2
meta_heuristic2.set_budget(100)

# Optimize the test function using the MetaHeuristic2
best_func2 = meta_heuristic2(__call__, 100)

# Print the best function found
print("Best function:", best_func2)
print("Best fitness:", best_func2.budget)

# Plot the best functions found
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.plot(best_func.budget, best_func)
plt.title("Best Function Found")
plt.xlabel("Budget")
plt.ylabel("Fitness")

plt.subplot(1, 2, 2)
plt.plot(best_func2.budget, best_func2)
plt.title("Best Function Found (Alternative)")
plt.xlabel("Budget")
plt.ylabel("Fitness")

plt.tight_layout()
plt.show()