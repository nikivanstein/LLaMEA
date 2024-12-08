# Description: Evolutionary Black Box Optimization using Differential Evolution with Adaptive Line Search
# Code: 
# ```python
import numpy as np
from scipy.optimize import differential_evolution
import copy

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

        # Apply adaptive line search to refine the solution
        self.population = self.apply_adaptive_line_search(self.population)

        # Return the optimized function and its value
        return func(self.population[0]), -func(self.population[0])

    def apply_adaptive_line_search(self, population):
        """
        Apply adaptive line search to refine the solution.

        Args:
            population (list): The current population.

        Returns:
            list: The refined population.
        """
        # Define the adaptive line search parameters
        alpha = 0.5
        beta = 1.5
        max_iter = 100

        # Initialize the best solution and its value
        best_solution = None
        best_value = float('-inf')

        # Iterate over the population
        for i in range(len(population)):
            # Get the current solution
            current_solution = population[i]

            # Evaluate the objective function for the current solution
            current_value = -func(current_solution)

            # Update the best solution and its value
            if current_value > best_value:
                best_solution = current_solution
                best_value = current_value

            # Update the adaptive line search parameters
            alpha *= 0.5
            if alpha < 0.1:
                alpha = 0.1

            # Update the population with the adaptive line search
            population = [copy.deepcopy(current_solution) + [np.random.uniform(-alpha, alpha, size=self.dim)] for j in range(len(population))]

        # Return the refined population
        return population

# Description: Evolutionary Black Box Optimization using Differential Evolution with Adaptive Line Search
# Code: 
# ```python
import numpy as np
from scipy.optimize import differential_evolution
import copy

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

        # Apply adaptive line search to refine the solution
        self.population = self.apply_adaptive_line_search(self.population)

        # Return the optimized function and its value
        return func(self.population[0]), -func(self.population[0])

# Description: Evolutionary Black Box Optimization using Differential Evolution with Adaptive Line Search
# Code: 
# ```python
# def apply_adaptive_line_search(self, population):
#     """
#     Apply adaptive line search to refine the solution.

#     Args:
#         population (list): The current population.

#     Returns:
#         list: The refined population.
#     """
#     # Define the adaptive line search parameters
#     alpha = 0.5
#     beta = 1.5
#     max_iter = 100

#     # Initialize the best solution and its value
#     best_solution = None
#     best_value = float('-inf')

#     # Iterate over the population
#     for i in range(len(population)):
#         # Get the current solution
#         current_solution = population[i]

#         # Evaluate the objective function for the current solution
#         current_value = -func(current_solution)

#         # Update the best solution and its value
#         if current_value > best_value:
#             best_solution = current_solution
#             best_value = current_value

#         # Update the adaptive line search parameters
#         alpha *= 0.5
#         if alpha < 0.1:
#             alpha = 0.1

#         # Update the population with the adaptive line search
#         population = [copy.deepcopy(current_solution) + [np.random.uniform(-alpha, alpha, size=self.dim)] for j in range(len(population))]

#     # Return the refined population
#     return population