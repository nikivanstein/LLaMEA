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

        # Initialize the population
        self.population = [x.copy() for x in x]

        # Initialize the fitness values
        self.fitness_values = [func_value]

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

            # Evaluate the fitness of the new individual
            fitness = func(x)
            self.fitness_values.append(fitness)

            # Update the population
            self.population.append(x)

            # Update the best solution
            if fitness < best_func_value:
                best_x = x
                best_func_value = fitness

        # Evaluate the best solution at the final search point
        func_value = func(best_x)

        print(f"Best solution: {best_x}, Best function value: {func_value}")

        return best_x, func_value

# Define a simple black box function
def func(x):
    return np.sin(x) + 0.1 * np.cos(2 * x) + 0.2 * np.sin(3 * x)

# Create an instance of the AdaptiveBBOO class
bboo = AdaptiveBBOO(1000, 10)

# Optimize the function with 1000 evaluations
bboo.optimize(func)

# One-line description with the main idea:
# Adaptive Black Box Optimization using Genetic Algorithm
# 
# The algorithm uses a population-based approach, where the population size is set to 1000 and the dimensionality is set to 10.
# The population is initialized with random points in the search space and then evolved using a genetic algorithm, where the fitness of each individual is evaluated and used to select the next generation.
# The algorithm stops when the budget is exhausted or when a better solution is found.
# 
# The probability of changing the individual lines of the selected solution to refine its strategy is set to 0.4.