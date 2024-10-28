# Description: AdaptiveBBOO Algorithm
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
        self.best_individual = None
        self.best_function_value = np.inf
        self.best_individual_fitness = 0

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
        self.best_individual = x
        self.best_function_value = func_value

        # Perform the metaheuristic search
        for _ in range(self.budget):
            # Evaluate the function at the current search point
            func_value = func(x)

            # If the function value is better than the best found so far, update the best solution
            if func_value > self.best_function_value:
                self.best_individual = x
                self.best_function_value = func_value

            # If the search space is exhausted, stop the algorithm
            if np.all(x >= self.search_space):
                break

            # Randomly perturb the search point
            x = x + np.random.uniform(-1, 1, self.dim)

        # Evaluate the best solution at the final search point
        func_value = func(self.best_individual)

        print(f"Best solution: {self.best_individual}, Best function value: {func_value}")

        # Update the best individual and its fitness
        self.best_individual_fitness = self.best_function_value
        self.best_individual = self.best_individual.copy()
        self.best_function_value = func_value

        # If the best function value is better than the current best, update the best solution
        if self.best_function_value < self.best_individual_fitness:
            self.best_individual = self.best_individual.copy()
            self.best_function_value = self.best_individual_fitness

        return self.best_individual, self.best_function_value

# Define a simple black box function
def func(x):
    return np.sin(x) + 0.1 * np.cos(2 * x) + 0.2 * np.sin(3 * x)

# Create an instance of the AdaptiveBBOO class
bboo = AdaptiveBBO()

# Optimize the function with 1000 evaluations
bboo.optimize(func)

# Print the best solution and its fitness
print(f"Best solution: {bboo.best_individual}, Best function value: {bboo.best_function_value}")