import numpy as np
from scipy.optimize import differential_evolution

class AdaptiveBlackBoxOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.func_evals = 0
        self.func_values = None

    def __call__(self, func):
        if self.func_values is None:
            self.func_evals = self.budget
            self.func_values = np.zeros(self.dim)
            for _ in range(self.func_evals):
                func(self.func_values)
        else:
            while self.func_evals > 0:
                idx = np.argmin(np.abs(self.func_values))
                self.func_values[idx] = func(self.func_values[idx])
                self.func_evals -= 1
                if self.func_evals == 0:
                    break

    def adaptive_differential_evolution(self, func, bounds):
        """
        Adaptive Black Box Optimization using Differential Evolution.

        This algorithm uses Differential Evolution to search the search space.
        It starts with an initial population of random points and evolves it
        using the given bounds.

        Args:
            func (function): The objective function to optimize.
            bounds (tuple): The bounds for the search space.

        Returns:
            result (tuple): The optimized point and the score.
        """
        # Initialize the population with random points in the search space
        population = np.random.uniform(bounds[0], bounds[1], (self.budget, self.dim))
        population = np.reshape(population, (self.budget, self.dim))

        # Evolve the population using Differential Evolution
        for _ in range(self.budget):
            # Calculate the fitness of each individual
            fitness = np.array([func(point) for point in population])

            # Calculate the selection probabilities
            probabilities = fitness / np.max(fitness)

            # Select the fittest individuals
            selected_indices = np.random.choice(self.budget, size=self.budget, p=probabilities)

            # Create a new population by combining the selected individuals
            new_population = np.concatenate((population[selected_indices], population[~selected_indices]))

            # Update the population
            population = new_population

        # Return the optimized point and the score
        result = np.min(population, axis=0)
        return result, np.mean(fitness)

# Description: Adaptive Black Box Optimization using Differential Evolution
# Code: 
# ```python
# import numpy as np
# import scipy.optimize as optimize

# class AdaptiveBlackBoxOptimizer:
#     def __init__(self, budget, dim):
#         self.budget = budget
#         self.dim = dim
#         self.func_evals = 0
#         self.func_values = None

#     def __call__(self, func):
#         if self.func_values is None:
#             self.func_evals = self.budget
#             self.func_values = np.zeros(self.dim)
#             for _ in range(self.func_evals):
#                 func(self.func_values)
#         else:
#             while self.func_evals > 0:
#                 idx = np.argmin(np.abs(self.func_values))
#                 self.func_values[idx] = func(self.func_values[idx])
#                 self.func_evals -= 1
#                 if self.func_evals == 0:
#                     break

#     def adaptive_differential_evolution(self, func, bounds):
#         """
#         Adaptive Black Box Optimization using Differential Evolution.

#         This algorithm uses Differential Evolution to search the search space.
#         It starts with an initial population of random points and evolves it
#         using the given bounds.

#         Args:
#             func (function): The objective function to optimize.
#             bounds (tuple): The bounds for the search space.

#         Returns:
#             result (tuple): The optimized point and the score.
#         """
#         # Initialize the population with random points in the search space
#         population = np.random.uniform(bounds[0], bounds[1], (self.budget, self.dim))
#         population = np.reshape(population, (self.budget, self.dim))

#         # Evolve the population using Differential Evolution
#         for _ in range(self.budget):
#             # Calculate the fitness of each individual
#             fitness = np.array([func(point) for point in population])

#             # Calculate the selection probabilities
#             probabilities = fitness / np.max(fitness)

#             # Select the fittest individuals
#             selected_indices = np.random.choice(self.budget, size=self.budget, p=probabilities)

#             # Create a new population by combining the selected individuals
#             new_population = np.concatenate((population[selected_indices], population[~selected_indices]))

#             # Update the population
#             population = new_population

#         # Return the optimized point and the score
#         result = np.min(population, axis=0)
#         return result, np.mean(fitness)

# # Example usage
# optimizer = AdaptiveBlackBoxOptimizer(budget=10, dim=5)
# func = lambda x: x**2
# result, score = optimizer(adaptive_differential_evolution(func, (-5, 5)))
# print("Optimized point:", result)
# print("Score:", score)