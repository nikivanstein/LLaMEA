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

            # Create a new population with a modified fitness function
            def modified_fitness(individual):
                fitness = func_values[individual]
                # Apply the novel mutation strategy
                mutation_probability = 0.05
                if random.random() < mutation_probability:
                    # Randomly select an index in the individual
                    idx = random.randint(0, self.dim - 1)
                    # Apply a mutation to the individual
                    individual[idx] += random.uniform(-0.1, 0.1)
                return fitness

            population = [modified_fitness(individual) for individual in new_population]

        # Return the optimized parameters and the optimized function value
        return population, func(population)

# One-line description
# Evolutionary Algorithm for Black Box Optimization using Genetic Programming with a Novel Mutation Strategy