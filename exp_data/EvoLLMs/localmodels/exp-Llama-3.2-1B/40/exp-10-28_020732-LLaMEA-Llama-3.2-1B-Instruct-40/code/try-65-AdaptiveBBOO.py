# Description: Adaptive Black Box Optimization using Genetic Algorithm
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

        # Select a new solution using the adaptive strategy
        if np.random.rand() < 0.4:
            # Refine the solution using linear interpolation
            new_x = np.interp(np.linspace(0, 1, self.dim), np.linspace(-5.0, 5.0, 2), [best_x[0], best_x[-1]])
            new_func_value = func(new_x)
            print(f"New solution: {new_x}, New function value: {new_func_value}")

        return best_x, func_value

# Define a simple black box function
def func(x):
    return np.sin(x) + 0.1 * np.cos(2 * x) + 0.2 * np.sin(3 * x)

# Create an instance of the AdaptiveBBOO class
bboo = AdaptiveBBO()

# Optimize the function with 1000 evaluations
bboo.optimize(func)

# Plot the function and its corresponding fitness values
plt.plot(np.linspace(-5.0, 5.0, 100), np.sin(np.linspace(-5.0, 5.0, 100)) + 0.1 * np.cos(2 * np.linspace(-5.0, 5.0, 100)) + 0.2 * np.sin(3 * np.linspace(-5.0, 5.0, 100)), 'b')
plt.plot(np.linspace(-5.0, 5.0, 100), func(np.linspace(-5.0, 5.0, 100)), 'r')
plt.show()