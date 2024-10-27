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

        # Create a queue to store the individuals
        queue = deque(population)

        # Evaluate the function for each individual in the population
        for _ in range(self.budget):
            # Evaluate the function for each individual in the population
            func_values = func(queue)

            # Select the fittest individuals based on the function values
            fittest_individuals = np.argsort(func_values)[::-1][:self.population_size // 2]

            # Create a new population by combining the fittest individuals
            new_population = np.concatenate([population[:fittest_individuals.size // 2], fittest_individuals[fittest_individuals.size // 2:]])

            # Replace the old population with the new population
            population = new_population

            # Add the new individuals to the queue
            queue.extend(new_population)

        # Return the optimized parameters and the optimized function value
        return population, func(population)

# Mutation strategy: swap two random elements in the individual
def mutate(individual, budget):
    """
    Apply the mutation strategy to an individual.

    Parameters:
    individual (numpy array): The individual to mutate.
    budget (int): The maximum number of mutations allowed.

    Returns:
    numpy array: The mutated individual.
    """
    # Create a list of indices to swap
    indices = random.sample(range(len(individual)), len(individual))

    # Swap the elements at the selected indices
    individual[indices[0]], individual[indices[1]] = individual[indices[1]], individual[indices[0]]

    # Check for invalid mutation
    if individual[indices[0]] < -5.0 or individual[indices[0]] > 5.0:
        individual[indices[0]] = 5.0
    if individual[indices[1]] < -5.0 or individual[indices[1]] > 5.0:
        individual[indices[1]] = 5.0

    # Return the mutated individual
    return individual

# Description: Evolutionary Algorithm for Black Box Optimization using Genetic Programming with a Novel Mutation Strategy
# Code: 
# ```python
# import numpy as np
# import random
# from collections import deque

# class BlackBoxOptimizer:
#     def __init__(self, budget, dim):
#         """
#         Initialize the BlackBoxOptimizer with a given budget and dimensionality.

#         Parameters:
#         budget (int): The maximum number of function evaluations allowed.
#         dim (int): The dimensionality of the search space.
#         """
#         self.budget = budget
#         self.dim = dim

#     def __call__(self, func):
#         """
#         Optimize the black box function `func` using the given budget and search space.

#         Parameters:
#         func (function): The black box function to optimize.

#         Returns:
#         tuple: The optimized parameters and the optimized function value.
#         """
#         # Initialize the population size
#         population_size = 100

#         # Initialize the population with random parameters
#         population = np.random.uniform(-5.0, 5.0, (population_size, self.dim))

#         # Create a queue to store the individuals
#         queue = deque(population)

#         # Evaluate the function for each individual in the population
#         for _ in range(self.budget):
#             # Evaluate the function for each individual in the population
#             func_values = func(queue)

#             # Select the fittest individuals based on the function values
#             fittest_individuals = np.argsort(func_values)[::-1][:self.population_size // 2]

#             # Create a new population by combining the fittest individuals
#             new_population = np.concatenate([population[:fittest_individuals.size // 2], fittest_individuals[fittest_individuals.size // 2:]])

#             # Replace the old population with the new population
#             population = new_population

#             # Add the new individuals to the queue
#             queue.extend(new_population)

#         # Mutation strategy: swap two random elements in the individual
#         def mutate(individual, budget):
#             """
#             Apply the mutation strategy to an individual.

#             Parameters:
#             individual (numpy array): The individual to mutate.
#             budget (int): The maximum number of mutations allowed.

#             Returns:
#             numpy array: The mutated individual.
#             """
#             # Create a list of indices to swap
#             indices = random.sample(range(len(individual)), len(individual))

#             # Swap the elements at the selected indices
#             individual[indices[0]], individual[indices[1]] = individual[indices[1]], individual[indices[0]]

#             # Check for invalid mutation
#             if individual[indices[0]] < -5.0 or individual[indices[0]] > 5.0:
#                 individual[indices[0]] = 5.0
#             if individual[indices[1]] < -5.0 or individual[indices[1]] > 5.0:
#                 individual[indices[1]] = 5.0

#             # Return the mutated individual
#             return individual

#         # Apply the mutation strategy
#         mutated_individuals = []
#         for _ in range(budget):
#             # Add a new individual to the queue
#             mutated_individuals.append(mutate(queue[0], 100))

#         # Return the optimized parameters and the optimized function value
#         return mutated_individuals, func(np.array(mutated_individuals))

# def main():
#     # Initialize the BlackBoxOptimizer
#     optimizer = BlackBoxOptimizer(1000, 10)

#     # Optimize the function
#     optimized_individuals, optimized_function = optimizer(__call__)

#     # Print the results
#     print("Optimized Parameters:", optimized_individuals)
#     print("Optimized Function:", optimized_function)

# main()