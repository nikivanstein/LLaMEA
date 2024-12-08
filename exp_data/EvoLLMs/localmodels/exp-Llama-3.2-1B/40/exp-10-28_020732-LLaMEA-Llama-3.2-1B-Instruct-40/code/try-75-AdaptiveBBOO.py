# Description: Adaptive Black Box Optimization using Evolutionary Strategies
# Code: 
# ```python
import numpy as np
import matplotlib.pyplot as plt

class AdaptiveBBOO:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.search_space = np.linspace(-5.0, 5.0, 100)
        self.func_evaluations = []

    def __call__(self, func):
        def func_wrapper(x):
            return func(x)
        self.func_evaluations.append(func_wrapper)
        return func_wrapper

    def _select_x(self, budget):
        x = np.random.choice(self.search_space, budget, replace=False)
        return x

    def _select_f(self, budget):
        f_values = [func(x) for x in self._select_x(budget)]
        return np.random.choice(f_values, budget, replace=False)

    def _select_x_perturb(self, budget):
        x = self._select_x(budget)
        x_perturbed = x + np.random.uniform(-1, 1, self.dim)
        return x_perturbed

    def _select_f_perturb(self, budget):
        f_values = [func(x) for x in self._select_x(budget)]
        f_perturbed = f_values + np.random.uniform(-1, 1, self.dim, size=budget)
        return f_perturbed

    def optimize(self, func):
        # Initialize the search space
        x = self._select_x(self.budget)

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
            x = self._select_x_perturb(self.budget)

            # Evaluate the function at the perturbed search point
            func_value = func(x)

            # If the function value is better than the best found so far, update the best solution
            if func_value > best_func_value:
                best_x = x
                best_func_value = func_value

        # Evaluate the best solution at the final search point
        func_value = func(best_x)

        print(f"Best solution: {best_x}, Best function value: {func_value}")

        return best_x, func_value

# Define a simple black box function
def func(x):
    return np.sin(x) + 0.1 * np.cos(2 * x) + 0.2 * np.sin(3 * x)

# Create an instance of the AdaptiveBBOO class
bboo = AdaptiveBBOO(1000, 5)

# Optimize the function with 1000 evaluations
bboo.optimize(func)

# One-line description with the main idea
# Adaptive Black Box Optimization using Evolutionary Strategies
# 
# This algorithm uses a population of candidate solutions, each represented by a perturbed version of the search space.
# It iteratively selects the best solution from the population, evaluates the function at the selected solution, and updates the population based on the fitness values.