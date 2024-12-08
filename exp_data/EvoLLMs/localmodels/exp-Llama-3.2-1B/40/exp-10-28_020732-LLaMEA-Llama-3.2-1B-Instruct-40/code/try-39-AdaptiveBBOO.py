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

            # Randomly perturb the search point using adaptive mutation
            x = x + np.random.uniform(-1, 1, self.dim) * np.exp(np.random.uniform(0, 1, self.dim))

        # Evaluate the best solution at the final search point
        func_value = func(best_x)

        print(f"Best solution: {best_x}, Best function value: {func_value}")

        return best_x, func_value

# Define a simple black box function
def func(x):
    return np.sin(x) + 0.1 * np.cos(2 * x) + 0.2 * np.sin(3 * x)

# Define a black box function to be optimized
def func_bboo(x):
    return np.sin(x) + 0.1 * np.cos(2 * x) + 0.2 * np.sin(3 * x)

# Create an instance of the AdaptiveBBOO class
bboo = AdaptiveBBO()

# Optimize the function with 1000 evaluations
bboo.optimize(func_bboo)

# Define a black box function to be optimized with a different search space
def func_bboo2(x):
    return np.sin(x) + 0.1 * np.cos(2 * x) + 0.2 * np.sin(3 * x)
# Create an instance of the AdaptiveBBOO class with the new search space
bboo2 = AdaptiveBBO(100, 5)

# Optimize the function with 1000 evaluations
bboo2.optimize(func_bboo2)

# Define a black box function to be optimized with a different search space
def func_bboo3(x):
    return np.sin(x) + 0.1 * np.cos(2 * x) + 0.2 * np.sin(3 * x)
# Create an instance of the AdaptiveBBOO class with the new search space
bboo3 = AdaptiveBBO(100, 5)

# Optimize the function with 1000 evaluations
bboo3.optimize(func_bboo3)

# Define a black box function to be optimized with a different search space
def func_bboo4(x):
    return np.sin(x) + 0.1 * np.cos(2 * x) + 0.2 * np.sin(3 * x)
# Create an instance of the AdaptiveBBOO class with the new search space
bboo4 = AdaptiveBBO(100, 5)

# Optimize the function with 1000 evaluations
bboo4.optimize(func_bboo4)