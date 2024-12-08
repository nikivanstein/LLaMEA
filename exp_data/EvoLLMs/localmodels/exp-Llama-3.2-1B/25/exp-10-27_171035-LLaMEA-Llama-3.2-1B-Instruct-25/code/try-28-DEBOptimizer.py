# Description: Evolutionary Black Box Optimization using Differential Evolution with Adaptive Line Search
# Code: 
# ```python
import numpy as np
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
        self.search_space = (-5.0, 5.0)
        self.population_size = 100
        self.num_generations = 100
        self.search_history = []
        self.adaptive_line_search = False

    def __call__(self, func):
        """
        Optimize a black box function using DEBOptimizer.

        Args:
            func (function): The black box function to optimize.

        Returns:
            tuple: A tuple containing the optimized function and its value.
        """
        # Get the bounds of the search space
        lower_bound = self.search_space[0]
        upper_bound = self.search_space[1]

        # Initialize the population with random solutions
        self.population = [np.random.uniform(lower_bound, upper_bound, size=(self.population_size, self.dim)) for _ in range(self.population_size)]

        # Evaluate the objective function for each individual in the population
        results = []
        for _ in range(self.num_generations):
            # Evaluate the objective function for each individual in the population
            fitness_values = differential_evolution(lambda x: -func(x), self.population, bounds=(lower_bound, upper_bound), x0=self.population)

            # Select the fittest individuals for the next generation
            fittest_individuals = [self.population[i] for i, _ in enumerate(results) if _ == fitness_values.x[0]]

            # Replace the least fit individuals with the fittest ones
            self.population = [self.population[i] for i in range(self.population_size) if i not in [j for j, _ in enumerate(results) if _ == fitness_values.x[0]]]

            # Update the population with the fittest individuals
            self.population += [self.population[i] for i in range(self.population_size) if i not in [j for j, _ in enumerate(results) if _ == fitness_values.x[0]]]

            # Check if the population has reached the budget
            if len(self.population) > self.budget:
                break

        # Calculate the adaptive line search
        if self.adaptive_line_search:
            # Get the current best individual
            best_individual = self.population[0]

            # Evaluate the objective function for the current best individual
            fitness_values = differential_evolution(lambda x: -func(x), [best_individual], bounds=(lower_bound, upper_bound), x0=best_individual)

            # Update the search space bounds
            self.search_space = (fitness_values.x[0] - 0.1, fitness_values.x[0] + 0.1)

        # Return the optimized function and its value
        return func(self.population[0]), -func(self.population[0])
