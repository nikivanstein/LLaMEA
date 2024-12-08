import numpy as np
import random
from scipy.optimize import minimize

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
            # Define the mutation strategy
            if random.random() < 0.05:
                # Randomly select a dimension and add a new value
                dim_idx = random.randint(0, self.dim - 1)
                individual[dim_idx] += np.random.uniform(-1.0, 1.0)
            return individual

        # Define the selection function
        def select(population):
            # Define the selection strategy
            return np.argsort(func(population))

        # Define the crossover function
        def crossover(parent1, parent2):
            # Define the crossover strategy
            if random.random() < 0.05:
                # Randomly select a dimension and split the parents
                dim_idx = random.randint(0, self.dim - 1)
                parent1[dim_idx] = random.uniform(-5.0, 5.0)
                parent2[dim_idx] = random.uniform(-5.0, 5.0)
            return np.concatenate([parent1, parent2])

        # Define the mutation function
        def mutate(individual):
            # Define the mutation strategy
            if random.random() < 0.05:
                # Randomly select a dimension and add a new value
                dim_idx = random.randint(0, self.dim - 1)
                individual[dim_idx] += np.random.uniform(-1.0, 1.0)
            return individual

        # Select the fittest individuals based on the function values
        fittest_individuals = select(population)

        # Create a new population by combining the fittest individuals
        new_population = np.concatenate([fittest_individuals[:fittest_individuals.size // 2], fittest_individuals[fittest_individuals.size // 2:]])

        # Replace the old population with the new population
        population = new_population

        # Evaluate the function for the new population
        func_values = func(population)

        # Select the fittest individuals based on the function values
        fittest_individuals = select(population)

        # Create a new population by combining the fittest individuals
        new_population = np.concatenate([fittest_individuals[:fittest_individuals.size // 2], fittest_individuals[fittest_individuals.size // 2:]])

        # Replace the old population with the new population
        population = new_population

        # Evaluate the function for the new population
        func_values = func(population)

        # Return the optimized parameters and the optimized function value
        return population, func(population), func_values

# Description: Evolutionary Algorithm for Black Box Optimization using Genetic Programming with a Novel Mutation Strategy
# Code: 