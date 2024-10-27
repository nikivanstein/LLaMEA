# Description: Evolutionary Black Box Optimization using Differential Evolution with Adaptive Line Search
# Code: 
# ```python
import numpy as np
from scipy.optimize import differential_evolution
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
        self.step_size = 1.0
        self.step_size_factor = 0.1
        self.iterations = 0
        self.best_solution = None
        self.best_fitness = float('-inf')

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

        # Update the best solution and fitness
        if len(self.population) > 0:
            self.best_solution = self.population[0]
            self.best_fitness = -func(self.best_solution)

        # Apply adaptive line search
        if self.iterations < 100:
            self.step_size *= 0.9
            self.step_size_factor *= 0.9
            self.iterations += 1

        # Return the optimized function and its value
        return func(self.best_solution), -func(self.best_solution)

DEBOptimizer(1000, 5).__call__(lambda x: -x)