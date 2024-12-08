# Description: Evolutionary Algorithm for Black Box Optimization using Genetic Programming with a Novel Mutation Strategy
# Code: 
# ```python
import numpy as np
import random
from collections import deque

class BlackBoxOptimizer:
    def __init__(self, budget, dim):
        """
        Initialize the BlackBoxOptimizer with a given budget and dimensionality.

        Parameters:
        budget (int): The maximum number of function evaluations allowed.
        dim (int): The dimensionality of the search space.
        """
        self.budget = budget
        self.dim = dim
        self.population_size = 100
        self.iterations = 1000
        self mutation_rate = 0.05

    def __call__(self, func):
        """
        Optimize the black box function `func` using the given budget and search space.

        Parameters:
        func (function): The black box function to optimize.

        Returns:
        tuple: The optimized parameters and the optimized function value.
        """
        # Initialize the population with random parameters
        population = np.random.uniform(-5.0, 5.0, (self.population_size, self.dim))

        # Evaluate the function for each individual in the population
        for _ in range(self.budget):
            # Evaluate the function for each individual in the population
            func_values = func(population)

            # Select the fittest individuals based on the function values
            fittest_individuals = np.argsort(func_values)[::-1][:self.population_size // 2]

            # Create a new population by combining the fittest individuals
            new_population = np.concatenate([population[:fittest_individuals.size // 2], fittest_individuals[fittest_individuals.size // 2:]])

            # Replace the old population with the new population
            population = new_population

            # Refine the strategy using the new population
            self.refine_strategy(population, func_values)

        # Return the optimized parameters and the optimized function value
        return population, func(population)

    def refine_strategy(self, population, func_values):
        """
        Refine the strategy using the new population.

        Parameters:
        population (numpy array): The new population.
        func_values (numpy array): The function values for the new population.
        """
        # Initialize the new population
        new_population = np.random.uniform(-5.0, 5.0, (self.population_size, self.dim))

        # Evaluate the function for each individual in the new population
        for _ in range(self.iterations):
            # Evaluate the function for each individual in the new population
            func_values_new = func_values

            # Select the fittest individuals based on the function values
            fittest_individuals_new = np.argsort(func_values_new)[::-1][:self.population_size // 2]

            # Create a new population by combining the fittest individuals
            new_population = np.concatenate([population[:fittest_individuals_new.size // 2], fittest_individuals_new[fittest_individuals_new.size // 2:]])

            # Replace the old population with the new population
            population = new_population

            # Update the strategy using the new population
            self.update_strategy(population, func_values_new)

        # Return the optimized parameters and the optimized function value
        return population, func_values_new

    def update_strategy(self, population, func_values):
        """
        Update the strategy using the new population.

        Parameters:
        population (numpy array): The new population.
        func_values (numpy array): The function values for the new population.
        """
        # Initialize the best individual and its function value
        best_individual = population[np.argmax(func_values)]
        best_function_value = func_values[np.argmax(func_values)]

        # Initialize the mutation rate
        mutation_rate = 0.05

        # Initialize the mutation counter
        mutation_counter = 0

        # Generate a new individual by mutation
        while mutation_counter < self.population_size // 2:
            # Generate a random individual
            individual = np.random.uniform(-5.0, 5.0, self.dim)

            # Evaluate the function for the new individual
            func_value = func(individual)

            # Check if the new individual is better than the best individual
            if func_value > best_function_value:
                # Update the best individual and its function value
                best_individual = individual
                best_function_value = func_value

            # Generate a random mutation
            mutation = np.random.uniform(-1.0, 1.0, self.dim)

            # Apply mutation to the individual
            mutated_individual = individual + mutation

            # Evaluate the function for the mutated individual
            func_value_mutated = func(mutated_individual)

            # Check if the mutated individual is better than the best individual
            if func_value_mutated > best_function_value:
                # Update the best individual and its function value
                best_individual = mutated_individual
                best_function_value = func_value_mutated

            # Increment the mutation counter
            mutation_counter += 1

        # Return the optimized parameters and the optimized function value
        return best_individual, best_function_value