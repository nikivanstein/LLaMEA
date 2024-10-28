# Description: Evolutionary Black Box Optimization using Differential Evolution with Adaptive Line Search
# Code: 
# ```python
import numpy as np
from scipy.optimize import differential_evolution
from scipy.optimize import differential_evolution as deb
from scipy.optimize import minimize
import random
import math

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
            fitness_values = deb(func, bounds=(lower_bound, upper_bound), x0=self.population, method="SLSQP", constraints={"type": "eq", "fun": lambda x: -x[0]}, options={"maxiter": 1000})
            fitness_values.x = [x for x in fitness_values.x if x[0] >= -5.0 and x[0] <= 5.0]
            fitness_values.fun = -fitness_values.fun  # Reflect the negative fitness values

            # Select the fittest individuals for the next generation
            fittest_individuals = [self.population[i] for i, _ in enumerate(results) if _ == fitness_values.x[0]]

            # Replace the least fit individuals with the fittest ones
            self.population = [self.population[i] for i in range(population_size) if i not in [j for j, _ in enumerate(results) if _ == fitness_values.x[0]]]

            # Update the population with the fittest individuals
            self.population += [self.population[i] for i in range(population_size) if i not in [j for j, _ in enumerate(results) if _ == fitness_values.x[0]]]

            # Check if the population has reached the budget
            if len(self.population) > self.budget:
                break

        # Return the optimized function and its value
        return func(self.population[0]), -func(self.population[0])

# One-line description with the main idea
# Evolutionary Black Box Optimization using Differential Evolution with Adaptive Line Search
# This algorithm optimizes a black box function by iteratively refining the search space using differential evolution and adaptive line search