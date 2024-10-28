import numpy as np
import matplotlib.pyplot as plt
import random

class AdaptiveBBOO:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.search_space = np.linspace(-5.0, 5.0, 100)
        self.func_evaluations = []
        self.refining_strategy = None

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

    def evolve(self, func):
        # Select the fittest individuals
        fittest_individuals = sorted(self.func_evaluations, key=self.func_evaluations.__getitem__, reverse=True)[:self.budget]

        # Refine the individuals based on their fitness
        for individual in fittest_individuals:
            # Apply the evolving strategy
            evolving_strategy = random.choice(self.refining_strategy)
            new_individual = evolving_strategy(individual, func)

            # Evaluate the new individual
            func_value = func(new_individual)

            # Update the individual and the best solution
            self.func_evaluations.append(func_wrapper(new_individual))
            if func_value > self.func_evaluations[-1][1]:
                best_x = new_individual
                best_func_value = func_value

        # Evaluate the best solution at the final search point
        func_value = func(best_x)

        print(f"Best solution: {best_x}, Best function value: {func_value}")

        return best_x, func_value

# Define a simple black box function
def func(x):
    return np.sin(x) + 0.1 * np.cos(2 * x) + 0.2 * np.sin(3 * x)

# Define the evolving strategy
def evolving_strategy(individual, func):
    # If the individual is a linear function, try to find a better linear function
    if isinstance(individual, np.ndarray):
        new_individual = individual + 0.1 * np.random.uniform(-1, 1, individual.shape[0])
        return new_individual

    # If the individual is a polynomial function, try to find a better polynomial function
    elif isinstance(individual, np.ndarray):
        new_individual = individual + 0.1 * np.random.uniform(-1, 1, individual.shape[0])
        coefficients = np.polyfit(individual, func(individual), 2)
        new_individual = np.poly1d(coefficients)
        return new_individual

# Create an instance of the AdaptiveBBOO class
bboo = AdaptiveBBOO(100, 10)

# Optimize the function with 1000 evaluations
bboo.optimize(func)

# Evolve the function with the evolving strategy
bboo.evolve(func)