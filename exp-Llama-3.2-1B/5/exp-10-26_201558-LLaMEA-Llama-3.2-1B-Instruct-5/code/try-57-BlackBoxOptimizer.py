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

            # Apply the mutation strategy
            for i in range(new_population.shape[0]):
                # Select a random individual from the new population
                individual = new_population[i]
                # Apply the mutation strategy: refine the individual lines
                if np.random.rand() < 0.05:
                    # Refine the individual lines using a new line of the form (x + ε, y + ε)
                    x, y = individual
                    if np.random.rand() < 0.5:
                        x += np.random.uniform(-0.1, 0.1)
                    if np.random.rand() < 0.5:
                        y += np.random.uniform(-0.1, 0.1)
                # Replace the old population with the new population
                population = new_population

        # Return the optimized parameters and the optimized function value
        return population, func(population)

# Description: Novel mutation strategy for evolutionary algorithm
# Code: 
# ```python
# import numpy as np
# import random
#
# class BlackBoxOptimizer:
#     def __init__(self, budget, dim):
#         """
#         Initialize the BlackBoxOptimizer with a given budget and dimensionality.
#
#         Parameters:
#         budget (int): The maximum number of function evaluations allowed.
#         dim (int): The dimensionality of the search space.
#         """
#         self.budget = budget
#         self.dim = dim
#
#     def __call__(self, func):
#         """
#         Optimize the black box function `func` using the given budget and search space.
#
#         Parameters:
#         func (function): The black box function to optimize.
#
#         Returns:
#         tuple: The optimized parameters and the optimized function value.
#         """
#         # Initialize the population size
#         population_size = 100

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

            # Apply the mutation strategy
            for i in range(new_population.shape[0]):
                # Select a random individual from the new population
                individual = new_population[i]
                # Apply the mutation strategy: refine the individual lines
                if np.random.rand() < 0.05:
                    # Refine the individual lines using a new line of the form (x + ε, y + ε)
                    x, y = individual
                    if np.random.rand() < 0.5:
                        x += np.random.uniform(-0.1, 0.1)
                    if np.random.rand() < 0.5:
                        y += np.random.uniform(-0.1, 0.1)
                # Replace the old population with the new population
                population = new_population

        # Return the optimized parameters and the optimized function value
        return population, func(population)