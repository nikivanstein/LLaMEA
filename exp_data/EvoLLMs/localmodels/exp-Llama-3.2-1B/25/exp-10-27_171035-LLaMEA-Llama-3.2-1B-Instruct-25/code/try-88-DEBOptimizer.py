import numpy as np
from scipy.optimize import differential_evolution

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
        self mutation_rate = 0.1

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

        # Evaluate the objective function for the last generation to get the final solution
        fitness_values = differential_evolution(lambda x: -func(x), self.population, bounds=(lower_bound, upper_bound), x0=self.population)
        final_individual = self.population[fitness_values.x[0]]
        final_individual = np.array(final_individual) / np.sqrt(len(final_individual))

        # Apply adaptive mutation to the final solution
        mutation_indices = np.random.choice(len(final_individual), self.budget, replace=False)
        mutation_indices = mutation_indices[:self.budget]
        final_individual[mutation_indices] += np.random.uniform(-1, 1, size=(self.budget, self.dim))

        # Return the optimized function and its value
        return func(final_individual), -func(final_individual)

# Description: Evolutionary Black Box Optimization using Differential Evolution with Adaptive Mutation Strategy
# Code: 
# ```python
# import numpy as np
# import random
# import scipy.optimize as optimize

# class DEBOptimizer:
#     def __init__(self, budget, dim):
#         self.budget = budget
#         self.dim = dim
#         self.func = None
#         self.mutation_rate = 0.1

#     def __call__(self, func):
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
#             fitness_values = optimize.differential_evolution(lambda x: -func(x), self.population, bounds=(lower_bound, upper_bound), x0=self.population)

#             # Select the fittest individuals for the next generation
#             fittest_individuals = [self.population[i] for i, _ in enumerate(results) if _ == fitness_values.x[0]]

#             # Replace the least fit individuals with the fittest ones
#             self.population = [self.population[i] for i in range(population_size) if i not in [j for j, _ in enumerate(results) if _ == fitness_values.x[0]]]

#             # Update the population with the fittest individuals
#             self.population += [self.population[i] for i in range(population_size) if i not in [j for j, _ in enumerate(results) if _ == fitness_values.x[0]]]

#             # Check if the population has reached the budget
#             if len(self.population) > self.budget:
#                 break

#         # Evaluate the objective function for the last generation to get the final solution
#         fitness_values = optimize.differential_evolution(lambda x: -func(x), self.population, bounds=(lower_bound, upper_bound), x0=self.population)
#         final_individual = self.population[fitness_values.x[0]]
#         final_individual = np.array(final_individual) / np.sqrt(len(final_individual))

#         # Apply adaptive mutation to the final solution
#         mutation_indices = np.random.choice(len(final_individual), self.budget, replace=False)
#         mutation_indices = mutation_indices[:self.budget]
#         final_individual[mutation_indices] += np.random.uniform(-1, 1, size=(self.budget, self.dim))

#         # Return the optimized function and its value
#         return func(final_individual), -func(final_individual)

# optimizer = DEBOptimizer(budget=100, dim=10)
# func = lambda x: x**2
# optimized_func, optimized_value = optimizer(func)
# print(f"Optimized function: {optimized_func}, Optimized value: {optimized_value}")