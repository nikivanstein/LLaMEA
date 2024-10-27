# Description: Evolutionary Algorithm for Black Box Optimization using Genetic Programming with a Novel Mutation Strategy
# Code: 
# ```python
import numpy as np
import random

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

    def __call__(self, func, mutation_rate):
        """
        Optimize the black box function `func` using the given budget and search space.

        Parameters:
        func (function): The black box function to optimize.
        mutation_rate (float): The probability of mutation.

        Returns:
        tuple: The optimized parameters and the optimized function value.
        """
        # Initialize the population size
        population_size = 100

        # Initialize the population with random parameters
        population = np.random.uniform(-5.0, 5.0, (population_size, self.dim))

        # Evaluate the function for each individual in the population
        for _ in range(self.budget):
            # Evaluate the function for each individual in the population
            func_values = func(population)

            # Select the fittest individuals based on the function values
            fittest_individuals = np.argsort(func_values)[::-1][:self.population_size // 2]

            # Create a new population by combining the fittest individuals
            new_population = np.concatenate([population[:fittest_individuals.size // 2], fittest_individuals[fittest_individuals.size // 2:]])

            # Apply mutation to the new population
            mutated_population = self.mutate(new_population, mutation_rate)

            # Replace the old population with the new population
            population = mutated_population

        # Return the optimized parameters and the optimized function value
        return population, func(population)

    def mutate(self, population, mutation_rate):
        """
        Apply mutation to the given population.

        Parameters:
        population (numpy.ndarray): The population to mutate.
        mutation_rate (float): The probability of mutation.

        Returns:
        numpy.ndarray: The mutated population.
        """
        # Create a copy of the population
        mutated_population = population.copy()

        # Randomly select individuals to mutate
        indices = np.random.choice(population.shape[0], size=population.shape[0], replace=False)

        # Apply mutation to the selected individuals
        for index in indices:
            # Generate a new individual by combining the mutated parent and a random child
            mutated_individual = np.concatenate([population[index], np.random.uniform(-5.0, 5.0, self.dim)])

            # Apply mutation to the new individual
            mutated_individual = self.applyMutation(mutated_individual, mutation_rate)

            # Replace the original individual with the mutated individual
            mutated_population[index] = mutated_individual

        return mutated_population

    def applyMutation(self, individual, mutation_rate):
        """
        Apply mutation to the given individual.

        Parameters:
        individual (numpy.ndarray): The individual to mutate.
        mutation_rate (float): The probability of mutation.

        Returns:
        numpy.ndarray: The mutated individual.
        """
        # Generate a new individual by applying mutation to the individual
        mutated_individual = individual.copy()
        for _ in range(self.dim):
            # Generate a random change
            change = np.random.uniform(-5.0, 5.0)

            # Apply the change to the individual
            mutated_individual += change

            # Ensure the individual stays within the search space
            mutated_individual = np.clip(mutated_individual, -5.0, 5.0)

        # Apply mutation with the given probability
        if random.random() < mutation_rate:
            # Generate a new change
            change = np.random.uniform(-5.0, 5.0)

            # Apply the change to the individual
            mutated_individual += change

            # Ensure the individual stays within the search space
            mutated_individual = np.clip(mutated_individual, -5.0, 5.0)

        return mutated_individual