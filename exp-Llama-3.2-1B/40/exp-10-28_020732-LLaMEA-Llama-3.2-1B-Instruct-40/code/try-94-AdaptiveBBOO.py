import numpy as np
import matplotlib.pyplot as plt
import random
from scipy.optimize import minimize

class AdaptiveBBOO:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.search_space = np.linspace(-5.0, 5.0, 100)
        self.func_evaluations = []
        self.refine_strategy = None

    def __call__(self, func):
        def func_wrapper(x):
            return func(x)
        self.func_evaluations.append(func_wrapper)
        return func_wrapper

    def optimize(self, func):
        # Initialize the search space
        x = self.search_space[np.random.choice(self.search_space.shape[0], self.budget)]

        # Perform the first evaluation
        func_value = func(x)
        print(f"Initial evaluation: {func_value}")

        # Initialize the best solution and best function value
        best_x = x
        best_func_value = func_value

        # Perform the metaheuristic search
        for _ in range(self.budget):
            # Evaluate the function at the current search point
            func_value = func(x)

            # If the function value is better than the best found so far, update the best solution
            if func_value > best_func_value:
                best_x = x
                best_func_value = func_value

            # If the search space is exhausted, stop the algorithm
            if np.all(x >= self.search_space):
                break

            # Randomly perturb the search point
            x = x + np.random.uniform(-1, 1, self.dim)

        # Evaluate the best solution at the final search point
        func_value = func(best_x)

        print(f"Best solution: {best_x}, Best function value: {func_value}")

        return best_x, func_value

    def refine_strategy(self, func, x, best_func_value, budget):
        # Refine the strategy based on the best function value and budget
        if budget > 0:
            # Select a new search point with a higher probability
            new_x = x + np.random.uniform(-1, 1, self.dim) * (best_func_value / func_value)
            # Evaluate the new search point
            new_func_value = func(new_x)
            # If the new function value is better than the current best, update the best solution
            if new_func_value > best_func_value:
                best_x = new_x
                best_func_value = new_func_value

            # If the search space is exhausted, stop the algorithm
            if np.all(new_x >= self.search_space):
                return best_x, best_func_value
        else:
            # If the budget is exhausted, return the current best solution
            return best_x, best_func_value

# Define a simple black box function
def func(x):
    return np.sin(x) + 0.1 * np.cos(2 * x) + 0.2 * np.sin(3 * x)

# Create an instance of the AdaptiveBBOO class
bboo = AdaptiveBBOO(1000, 10)

# Optimize the function with 1000 evaluations
bboo.optimize(func)

# Define a new strategy
def new_strategy(func, x, best_func_value, budget):
    # Refine the strategy based on the best function value and budget
    refined_x, refined_func_value = bboo.refine_strategy(func, x, best_func_value, budget)
    return refined_x, refined_func_value

# Optimize the function with the new strategy
bboo2 = AdaptiveBBOO(1000, 10)
bboo2.optimize(func)

# Define a simple black box function
def func2(x):
    return np.sin(x) + 0.1 * np.cos(2 * x) + 0.2 * np.sin(3 * x)

# Create an instance of the AdaptiveBBOO class
bboo3 = AdaptiveBBOO(1000, 10)

# Optimize the function with 1000 evaluations
bboo3.optimize(func2)

# Print the results
print(f"Bboo: Best solution: {bboo.best_x}, Best function value: {bboo.best_func_value}")
print(f"Bboo2: Best solution: {bboo2.best_x}, Best function value: {bboo2.best_func_value}")
print(f"Bboo3: Best solution: {bboo3.best_x}, Best function value: {bboo3.best_func_value}")