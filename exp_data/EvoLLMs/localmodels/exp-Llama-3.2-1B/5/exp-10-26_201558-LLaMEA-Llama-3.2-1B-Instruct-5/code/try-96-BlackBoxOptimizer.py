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

            # Apply a novel mutation strategy
            mutated_individuals = np.copy(new_population)
            mutated_individuals[np.random.randint(0, new_population.shape[0], self.population_size // 2)] = np.random.uniform(-5.0, 5.0, new_population.shape[1])
            mutated_individuals = np.sort(mutated_individuals, axis=0)  # Sort by each individual
            mutated_individuals = np.concatenate([mutated_individuals[:fittest_individuals.size // 2], mutated_individuals[fittest_individuals.size // 2:]], axis=0)

            # Replace the old population with the new population
            population = mutated_individuals

        # Return the optimized parameters and the optimized function value
        return population, func(population)

# Description: Evolutionary Algorithm for Black Box Optimization using Genetic Programming with a Novel Mutation Strategy
# Code: 
# ```python
# import numpy as np
# import random

class GeneticBlackBoxOptimizer:
    def __init__(self, budget, dim):
        """
        Initialize the GeneticBlackBoxOptimizer with a given budget and dimensionality.

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

            # Apply a novel mutation strategy
            mutated_individuals = np.copy(new_population)
            for i in range(population_size):
                if random.random() < 0.05:  # Refine strategy with a 5% probability
                    mutated_individuals[i] = np.random.uniform(-5.0, 5.0, self.dim)

            # Replace the old population with the new population
            population = mutated_individuals

        # Return the optimized parameters and the optimized function value
        return population, func(population)