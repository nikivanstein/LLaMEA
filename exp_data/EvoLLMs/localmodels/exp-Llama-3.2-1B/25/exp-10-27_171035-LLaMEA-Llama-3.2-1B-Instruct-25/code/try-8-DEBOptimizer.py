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

        # Return the optimized function and its value
        return func(self.population[0]), -func(self.population[0])

    def adaptiveMutation(self, individual):
        """
        Apply adaptive mutation to the individual.

        Args:
            individual (np.ndarray): The individual to mutate.

        Returns:
            np.ndarray: The mutated individual.
        """
        # Get the bounds of the search space
        lower_bound = -5.0
        upper_bound = 5.0

        # Randomly select a mutation point
        mutation_point = random.randint(0, self.dim - 1)

        # Generate a new individual by flipping the bit at the mutation point
        mutated_individual = individual.copy()
        mutated_individual[mutation_point] = 1 - mutated_individual[mutation_point]

        # Apply the mutation to the new individual
        mutated_individual[mutation_point] = 1 - mutated_individual[mutation_point]

        return mutated_individual

# Description: Evolutionary Black Box Optimization using Differential Evolution with Adaptive Mutation Strategy
# Code: 
# ```python
# import numpy as np
# from scipy.optimize import differential_evolution
# import random

# class DEBOptimizer:
#     def __init__(self, budget, dim):
#         """
#         Initialize the DEBOptimizer with a given budget and dimensionality.

#         Args:
#             budget (int): The maximum number of function evaluations allowed.
#             dim (int): The dimensionality of the search space.
#         """
#         self.budget = budget
#         self.dim = dim
#         self.func = None

#     def __call__(self, func):
#         """
#         Optimize a black box function using DEBOptimizer.

#         Args:
#             func (function): The black box function to optimize.

#         Returns:
#             tuple: A tuple containing the optimized function and its value.
#         """
#         # Get the bounds of the search space
#         lower_bound = -5.0
#         upper_bound = 5.0

#         # Initialize the population size and the number of generations
#         population_size = 100
#         num_generations = 100

#         # Initialize the population with random solutions
#         self.population = [np.random.uniform(lower_bound, upper_bound, size=(population_size, self.dim)) for _ in range(population_size)]

#         # Evaluate the objective function for each individual in the population
#         results = []
#         for _ in range(num_generations):
#             # Evaluate the objective function for each individual in the population
#             fitness_values = differential_evolution(lambda x: -func(x), self.population, bounds=(lower_bound, upper_bound), x0=self.population)

#             # Select the fittest individuals for the next generation
#             fittest_individuals = [self.population[i] for i, _ in enumerate(results) if _ == fitness_values.x[0]]

#             # Replace the least fit individuals with the fittest ones
#             self.population = [self.population[i] for i in range(population_size) if i not in [j for j, _ in enumerate(results) if _ == fitness_values.x[0]]]

#             # Update the population with the fittest individuals
#             self.population += [self.population[i] for i in range(population_size) if i not in [j for j, _ in enumerate(results) if _ == fitness_values.x[0]]]

#             # Check if the population has reached the budget
#             if len(self.population) > self.budget:
#                 break

#         # Return the optimized function and its value
#         return func(self.population[0]), -func(self.population[0])

    def adaptiveMutation(self, individual):
        """
        Apply adaptive mutation to the individual.

        Args:
            individual (np.ndarray): The individual to mutate.

        Returns:
            np.ndarray: The mutated individual.
        """
        # Get the bounds of the search space
        lower_bound = -5.0
        upper_bound = 5.0

        # Randomly select a mutation point
        mutation_point = random.randint(0, self.dim - 1)

        # Generate a new individual by flipping the bit at the mutation point
        mutated_individual = individual.copy()
        mutated_individual[mutation_point] = 1 - mutated_individual[mutation_point]

        # Apply the mutation to the new individual
        mutated_individual[mutation_point] = 1 - mutated_individual[mutation_point]

        return mutated_individual