# Description: Evolutionary Black Box Optimization using Differential Evolution with Adaptive Line Search
# Code: 
# ```python
import numpy as np
import scipy.optimize as optimize
from scipy.optimize import differential_evolution
import random

class DEBOptimizer:
    def __init__(self, budget, dim):
        """
        Initialize the DEBOptimizer with a given budget and dimensionality.

        Args:
            budget (int): The maximum number of function evaluations allowed.
            dim (int): The dimensionality of the search space.
        """
        self.budget = budget
        self.dim = dim
        self.func = None
        self.alpha = 0.25  # Probability of changing the line search direction

    def __call__(self, func):
        """
        Optimize a black box function using DEBOptimizer.

        Args:
            func (function): The black box function to optimize.

        Returns:
            tuple: A tuple containing the optimized function and its value.
        """
        # Get the bounds of the search space
        lower_bound = -5.0
        upper_bound = 5.0

        # Initialize the population size and the number of generations
        population_size = 100
        num_generations = 100

        # Initialize the population with random solutions
        self.population = [np.random.uniform(lower_bound, upper_bound, size=(population_size, self.dim)) for _ in range(population_size)]

        # Evaluate the objective function for each individual in the population
        results = []
        for _ in range(num_generations):
            # Evaluate the objective function for each individual in the population
            fitness_values = differential_evolution(lambda x: -func(x), self.population, bounds=(lower_bound, upper_bound), x0=self.population)

            # Select the fittest individuals for the next generation
            fittest_individuals = [self.population[i] for i, _ in enumerate(results) if _ == fitness_values.x[0]]

            # Replace the least fit individuals with the fittest ones
            self.population = [self.population[i] for i in range(population_size) if i not in [j for j, _ in enumerate(results) if _ == fitness_values.x[0]]]

            # Update the population with the fittest individuals
            self.population += [self.population[i] for i in range(population_size) if i not in [j for j, _ in enumerate(results) if _ == fitness_values.x[0]]]

            # Check if the population has reached the budget
            if len(self.population) > self.budget:
                break

        # Define the line search function
        def line_search(x, f, step_size):
            return f(x + step_size) - f(x)

        # Define the adaptive line search function
        def adaptive_line_search(x, f, step_size, alpha):
            # Calculate the step size
            step_size = alpha * np.sqrt(f(x))

            # Perform the line search
            new_x = x + step_size
            if np.abs(line_search(new_x, f, step_size)) < 1e-6:  # Stop when the step size is small
                return new_x

            # Update the step size
            step_size *= 0.9

            # Perform the line search again
            new_x = x + step_size
            if np.abs(line_search(new_x, f, step_size)) < 1e-6:  # Stop when the step size is small
                return new_x

            # Return the new x
            return new_x

        # Update the population with the adaptive line search
        for _ in range(population_size):
            new_individual = adaptive_line_search(self.population[0], func, self.population[0], self.alpha)
            self.population[0] = new_individual

        # Return the optimized function and its value
        return func(self.population[0]), -func(self.population[0])

# Example usage:
optimizer = DEBOptimizer(100, 10)
optimized_function, _ = optimizer(func)
print("Optimized function:", optimized_function)
print("Optimized value:", -optimized_function(optimized_function))