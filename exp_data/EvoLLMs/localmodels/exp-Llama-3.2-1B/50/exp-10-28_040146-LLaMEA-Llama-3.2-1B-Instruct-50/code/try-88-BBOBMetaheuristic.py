# Description: A novel metaheuristic algorithm that uses a random search with bounds to optimize black box functions.
# Code: 
# ```python
# import numpy as np
# import scipy.optimize as optimize
#
# class BBOBMetaheuristic:
#     def __init__(self, budget: int, dim: int):
#         """
#         Initialize the algorithm with a given budget and dimensionality.
#
#         Args:
#         budget (int): The maximum number of function evaluations allowed.
#         dim (int): The dimensionality of the search space.
#         """
#         self.budget = budget
#         self.dim = dim

#     def __call__(self, func: Any, bounds: Dict[str, float] = None) -> Any:
#         """
#         Optimize the given black box function using the provided bounds.

#         Args:
#         func (Any): The black box function to optimize.
#         bounds (Dict[str, float], optional): The bounds for the search space. Defaults to None.

#         Returns:
#         Any: The optimized function value.
#         """
#         # Create a copy of the function to avoid modifying the original function
#         func_copy = func.copy()

#         # Initialize the minimum and maximum function values
#         min_func = np.inf
#         max_func = -np.inf

#         # Iterate over the budget iterations
#         for _ in range(self.budget):
#             # Generate random function values within the bounds
#             random_func = np.random.uniform(-bounds["lower_bound"], bounds["upper_bound"], (self.dim,))

#             # Evaluate the function using the random values
#             func_value = func_copy(random_func)

#             # Update the minimum and maximum function values
#             min_func = min(min_func, func_value)
#             max_func = max(max_func, func_value)

#         # Return the optimized function value
#         return min_func

#     def evaluate_fitness(self, individual: np.ndarray) -> float:
#         """
#         Evaluate the fitness of a given individual.

#         Args:
#         individual (np.ndarray): The individual to evaluate.

#         Returns:
#         float: The fitness value.
#         """
#         # Call the original function with the individual
#         return self.f(individual)