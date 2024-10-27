# Description: Evolutionary Algorithm for Black Box Optimization using Genetic Programming with a Novel Mutation Strategy
# Code: 
# ```python
import numpy as np

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

            # Apply a mutation strategy to refine the individual lines
            mutated_individuals = np.concatenate([self.mutate(individual, self.dim, population_size) for individual in new_population])

            # Replace the old population with the new population
            population = mutated_individuals

        # Return the optimized parameters and the optimized function value
        return population, func(population)

    def mutate(self, individual, dim, population_size):
        """
        Apply a mutation strategy to refine the individual lines.

        Parameters:
        individual (numpy array): The individual to mutate.
        dim (int): The dimensionality of the search space.
        population_size (int): The size of the population.

        Returns:
        numpy array: The mutated individual.
        """
        # Generate a mutation vector
        mutation_vector = np.random.uniform(-1, 1, (dim,))

        # Mutate the individual using the mutation vector
        mutated_individual = individual + mutation_vector

        # Clip the mutated individual to the search space
        mutated_individual = np.clip(mutated_individual, -5.0, 5.0)

        # Return the mutated individual
        return mutated_individual