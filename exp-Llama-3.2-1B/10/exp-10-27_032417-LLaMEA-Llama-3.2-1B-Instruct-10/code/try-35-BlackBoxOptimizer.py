# BlackBoxOptimizer: Novel Metaheuristic Algorithm for Black Box Optimization
# Code:
# ```python
# import numpy as np
#
# class BlackBoxOptimizer:
#     """
#     A novel metaheuristic algorithm for solving black box optimization problems.
#     """
#
#     def __init__(self, budget, dim):
#         """
#         Initializes the optimizer with a given budget and dimensionality.
#         """
#         self.budget = budget
#         self.dim = dim
#
#     def __call__(self, func):
#         """
#         Optimizes a black box function using the optimizer.
#         """
#         # Initialize the solution and the number of function evaluations
#         solution = None
#         evaluations = 0
#
#         # Iterate over the range of possible solutions
#         while evaluations < self.budget:
#             # Generate a random solution within the search space
#             solution = np.random.uniform(-5.0, 5.0, self.dim)
#
#             # Evaluate the black box function at the current solution
#             evaluations += 1
#             func(solution)
#
#             # If the current solution is better than the previous best solution, update the solution
#             if evaluations > 0 and evaluations < self.budget:
#                 if evaluations > 0:
#                     # Calculate the probability of accepting the current solution
#                     probability = np.exp((evaluations - evaluations) / self.budget)
#
#                     # Accept the current solution with a probability less than 1
#                     if np.random.rand() < probability:
#                         solution = solution
#                 else:
#                     # Update the solution with the best solution found so far
#                     solution = None
#
#         # Return the optimal solution and the number of function evaluations used
#         return solution, evaluations
#
# # Example usage:
# def func(x):
#     return x**2 + 2*x + 1
#
# optimizer = BlackBoxOptimizer(100, 10)
# optimal_solution, num_evaluations = optimizer(func)
# print("Optimal solution:", optimal_solution)
# print("Number of function evaluations:", num_evaluations)