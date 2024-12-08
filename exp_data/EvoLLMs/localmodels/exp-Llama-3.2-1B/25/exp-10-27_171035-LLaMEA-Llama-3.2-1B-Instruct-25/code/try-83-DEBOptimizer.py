# Description: Evolutionary Black Box Optimization using Differential Evolution with Adaptive Mutation Strategy
# Code: 
# ```python
import numpy as np
import random
from scipy.optimize import differential_evolution

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
        self.population = None
        self.fitness_values = None
        self.mutation_rate = 0.1
        self.population_size = 100
        self.num_generations = 100
        self mutation_history = []

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

        # Initialize the population with random solutions
        self.population = [np.random.uniform(lower_bound, upper_bound, size=(self.population_size, self.dim)) for _ in range(self.population_size)]

        # Evaluate the objective function for each individual in the population
        self.fitness_values = []
        for _ in range(self.num_generations):
            # Evaluate the objective function for each individual in the population
            fitness_values = differential_evolution(lambda x: -func(x), self.population, bounds=(lower_bound, upper_bound), x0=self.population)

            # Select the fittest individuals for the next generation
            fittest_individuals = [self.population[i] for i, _ in enumerate(fitness_values.x) if fitness_values.x[0] == max(fitness_values.x)]

            # Replace the least fit individuals with the fittest ones
            self.population = [self.population[i] for i in range(self.population_size) if i not in [j for j, _ in enumerate(fitness_values.x) if fitness_values.x[0] == max(fitness_values.x)]] + fittest_individuals

            # Update the population with the fittest individuals
            self.population += [self.population[i] for i in range(self.population_size) if i not in [j for j, _ in enumerate(fitness_values.x) if fitness_values.x[0] == max(fitness_values.x)]] + fittest_individuals

            # Check if the population has reached the budget
            if len(self.population) > self.budget:
                break

        # Return the optimized function and its value
        if self.population:
            return func(self.population[0]), -func(self.population[0])
        else:
            return func(self.population[0]), -func(self.population[0])

# One-line description with the main idea
# Description: Evolutionary Black Box Optimization using Differential Evolution with Adaptive Mutation Strategy
# Code: 
# ```python
# ```python