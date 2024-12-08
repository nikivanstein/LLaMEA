import numpy as np
from scipy.optimize import differential_evolution

class AdaptiveDEBOptimizer:
    def __init__(self, budget, dim):
        """
        Initialize the AdaptiveDEBOptimizer with a given budget and dimensionality.

        Args:
            budget (int): The maximum number of function evaluations allowed.
            dim (int): The dimensionality of the search space.
        """
        self.budget = budget
        self.dim = dim
        self.func = None
        self.population = None
        self.fitness_values = None
        self.mutation_rate = 0.01
        self mutation_history = []

    def __call__(self, func):
        """
        Optimize a black box function using AdaptiveDEBOptimizer.

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
        self.fitness_values = differential_evolution(lambda x: -func(x), self.population, bounds=(lower_bound, upper_bound), x0=self.population)

        # Select the fittest individuals for the next generation
        self.population = [self.population[i] for i in range(population_size) if i not in [j for j, _ in enumerate(self.fitness_values.x) if _ == self.fitness_values.x[0]]]

        # Replace the least fit individuals with the fittest ones
        self.population = [self.population[i] for i in range(population_size) if i not in [j for j, _ in enumerate(self.fitness_values.x) if _ == self.fitness_values.x[0]]]

        # Update the population with the fittest individuals
        self.population += [self.population[i] for i in range(population_size) if i not in [j for j, _ in enumerate(self.fitness_values.x) if _ == self.fitness_values.x[0]]]

        # Check if the population has reached the budget
        if len(self.population) > self.budget:
            break

        # Update the mutation rate based on the fitness values
        if self.fitness_values.x[0] == -np.inf:
            self.mutation_rate = 0.1
        elif self.fitness_values.x[0] == np.inf:
            self.mutation_rate = 0.05
        else:
            self.mutation_rate = 0.01

        # Return the optimized function and its value
        return func(self.population[0]), -func(self.population[0])

# Description: Black Box Optimization using Evolutionary Differential Evolution with Adaptive Mutation
# Code: 
# ```python
# import numpy as np
# from scipy.optimize import differential_evolution
#
# class AdaptiveDEBOptimizer:
#     def __init__(self, budget, dim):
#         """
#         Initialize the AdaptiveDEBOptimizer with a given budget and dimensionality.
#         """
#         self.budget = budget
#         self.dim = dim
#         self.func = None
#         self.population = None
#         self.fitness_values = None
#         self.mutation_rate = 0.01
#         self.mutation_history = []
#
#     def __call__(self, func):
#         """
#         Optimize a black box function using AdaptiveDEBOptimizer.
#         """
#         # Get the bounds of the search space
#         lower_bound = -5.0
#         upper_bound = 5.0
#
#         # Initialize the population size and the number of generations
#         population_size = 100
#         num_generations = 100
#
#         # Initialize the population with random solutions
#         self.population = [np.random.uniform(lower_bound, upper_bound, size=(population_size, self.dim)) for _ in range(population_size)]
#
#         # Evaluate the objective function for each individual in the population
#         self.fitness_values = differential_evolution(lambda x: -func(x), self.population, bounds=(lower_bound, upper_bound), x0=self.population)
#
#         # Select the fittest individuals for the next generation
#         self.population = [self.population[i] for i in range(population_size) if i not in [j for j, _ in enumerate(self.fitness_values.x) if _ == self.fitness_values.x[0]]]
#
#         # Replace the least fit individuals with the fittest ones
#         self.population = [self.population[i] for i in range(population_size) if i not in [j for j, _ in enumerate(self.fitness_values.x) if _ == self.fitness_values.x[0]]]
#
#         # Update the population with the fittest individuals
#         self.population += [self.population[i] for i in range(population_size) if i not in [j for j, _ in enumerate(self.fitness_values.x) if _ == self.fitness_values.x[0]]]
#
#         # Check if the population has reached the budget
#         if len(self.population) > self.budget:
#             break
#
#         # Update the mutation rate based on the fitness values
#         if self.fitness_values.x[0] == -np.inf:
#             self.mutation_rate = 0.1
#         elif self.fitness_values.x[0] == np.inf:
#             self.mutation_rate = 0.05
#         else:
#             self.mutation_rate = 0.01
#
#         # Return the optimized function and its value
#         return func(self.population[0]), -func(self.population[0])