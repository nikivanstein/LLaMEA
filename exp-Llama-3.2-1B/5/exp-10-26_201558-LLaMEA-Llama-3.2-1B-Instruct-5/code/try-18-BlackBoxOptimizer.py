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

    def __call__(self, func):
        """
        Optimize the black box function `func` using the given budget and search space.

        Parameters:
        func (function): The black box function to optimize.

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

            # Replace the old population with the new population
            population = new_population

        # Define the mutation function
        def mutate(individual):
            if random.random() < 0.05:
                # Randomly select an individual from the population
                parent1, parent2 = random.sample(population, 2)

                # Select a random point within the search space
                point = np.random.uniform(-5.0, 5.0)

                # Create a new individual by combining the parent1 and parent2 with the point
                child = np.concatenate((parent1, [point]), axis=0)

                return child

        # Define the crossover function
        def crossover(parent1, parent2):
            # Select a random point within the search space
            point = np.random.uniform(-5.0, 5.0)

            # Create a new individual by combining the parent1 and parent2 with the point
            child = np.concatenate((parent1, [point]), axis=0)

            return child

        # Define the selection function
        def selection(population):
            # Select the fittest individuals based on the function values
            fittest_individuals = np.argsort(func_values)[::-1][:self.population_size // 2]

            # Create a new population by combining the fittest individuals
            new_population = np.concatenate([population[:fittest_individuals.size // 2], fittest_individuals[fittest_individuals.size // 2:]])

            return new_population

        # Define the evolution function
        def evolution(population):
            # Select the fittest individuals
            population = selection(population)

            # Perform crossover and mutation on the fittest individuals
            for _ in range(self.budget):
                # Select two parents
                parent1, parent2 = random.sample(population, 2)

                # Perform crossover
                child = crossover(parent1, parent2)

                # Perform mutation
                child = mutate(child)

                # Replace the old population with the new population
                population = np.concatenate([population[:fittest_individuals.size // 2], fittest_individuals[fittest_individuals.size // 2:]])

            return population

        # Optimize the function using the evolution function
        return evolution(population)

# Description: Evolutionary Algorithm for Black Box Optimization using Genetic Programming with a Novel Mutation Strategy
# Code: 