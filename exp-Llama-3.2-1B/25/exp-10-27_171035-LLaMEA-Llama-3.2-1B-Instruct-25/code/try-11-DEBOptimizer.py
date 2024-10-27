# Description: Evolutionary Black Box Optimization using Differential Evolution with Adaptive Mutation
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

        # Define the mutation strategy
        def mutation(individual, mutation_rate):
            """
            Apply mutation to an individual.

            Args:
                individual (list): The individual to mutate.
                mutation_rate (float): The probability of mutation.

            Returns:
                list: The mutated individual.
            """
            # Select a random index
            idx = random.randint(0, len(individual) - 1)

            # Apply mutation
            mutated_individual = individual.copy()
            mutated_individual[idx] = random.uniform(lower_bound, upper_bound)

            return mutated_individual

        # Apply mutation to the population
        for _ in range(population_size):
            mutated_individual = mutation(self.population[_], 0.1)
            self.population[_] = mutated_individual

        # Return the optimized function and its value
        return func(self.population[0]), -func(self.population[0])

# One-line description
# Evolutionary Black Box Optimization using Differential Evolution with Adaptive Mutation
# 