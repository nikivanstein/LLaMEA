import random
import numpy as np
from scipy.optimize import minimize
from collections import deque

class BlackBoxOptimizer:
    def __init__(self, budget, dim):
        """
        Initialize the BlackBoxOptimizer with a budget and dimensionality.

        Args:
            budget (int): The maximum number of function evaluations allowed.
            dim (int): The dimensionality of the search space.
        """
        self.budget = budget
        self.dim = dim
        self.search_space = np.linspace(-5.0, 5.0, dim)

    def __call__(self, func, initial_individual=None):
        """
        Optimize the black box function using the BlackBoxOptimizer.

        Args:
            func (callable): The black box function to optimize.
            initial_individual (list, optional): The initial individual for the optimization process. Defaults to None.

        Returns:
            float: The optimized value of the function.
        """
        # If no initial individual is provided, generate a random one
        if initial_individual is None:
            initial_individual = self.search_space[np.random.randint(0, self.dim)]

        # Initialize the best value and its corresponding index
        best_value = float('-inf')
        best_index = -1

        # Perform the specified number of function evaluations
        for _ in range(self.budget):
            # Generate a random point in the search space
            point = self.search_space[np.random.randint(0, self.dim)]

            # Evaluate the function at the current point
            value = func(point)

            # If the current value is better than the best value found so far,
            # update the best value and its corresponding index
            if value > best_value:
                best_value = value
                best_index = point

        # Return the optimized value
        return best_value

def __call__(self, func, initial_individual, budget, max_iter):
    """
    Optimize the black box function using the BlackBoxOptimizer.

    Args:
        func (callable): The black box function to optimize.
        initial_individual (list): The initial individual for the optimization process.
        budget (int): The maximum number of function evaluations allowed.
        max_iter (int): The maximum number of iterations for the optimization process.

    Returns:
        float: The optimized value of the function.
    """
    # Initialize the best value and its corresponding index
    best_value = float('-inf')
    best_index = -1

    # Initialize the current individual
    current_individual = initial_individual

    # Initialize the population of individuals
    population = deque([current_individual])

    # Perform the specified number of iterations
    for _ in range(max_iter):
        # Evaluate the function at the current individual
        value = func(current_individual)

        # If the current value is better than the best value found so far,
        # update the best value and its corresponding index
        if value > best_value:
            best_value = value
            best_index = current_individual
        else:
            # If the current value is not better than the best value found so far,
            # replace the current individual with the best individual
            best_individual = best_index
            best_value = best_value
            best_index = np.random.randint(0, self.dim)

        # Add the best individual to the population
        population.append(best_individual)

        # If the maximum number of iterations is reached,
        # replace the current individual with the best individual
        if len(population) > self.budget:
            current_individual = population.popleft()

    # Return the optimized value
    return best_value

# Description: Novel Metaheuristic Algorithm for Black Box Optimization
# Code: 
# ```python
# import random
# import numpy as np
# from scipy.optimize import minimize
# from collections import deque

# class BlackBoxOptimizer:
#     def __init__(self, budget, dim):
#         """
#         Initialize the BlackBoxOptimizer with a budget and dimensionality.

#         Args:
#             budget (int): The maximum number of function evaluations allowed.
#             dim (int): The dimensionality of the search space.
#         """
#         self.budget = budget
#         self.dim = dim
#         self.search_space = np.linspace(-5.0, 5.0, dim)

#     def __call__(self, func, initial_individual=None):
#         """
#         Optimize the black box function using the BlackBoxOptimizer.

#         Args:
#             func (callable): The black box function to optimize.
#             initial_individual (list, optional): The initial individual for the optimization process. Defaults to None.

#         Returns:
#             float: The optimized value of the function.
#         """
#         # If no initial individual is provided, generate a random one
#         if initial_individual is None:
#             initial_individual = self.search_space[np.random.randint(0, self.dim)]

#         # Initialize the best value and its corresponding index
#         best_value = float('-inf')
#         best_index = -1

#         # Perform the specified number of function evaluations
#         for _ in range(self.budget):
#             # Generate a random point in the search space
#             point = self.search_space[np.random.randint(0, self.dim)]

#             # Evaluate the function at the current point
#             value = func(point)

#             # If the current value is better than the best value found so far,
#             # update the best value and its corresponding index
#             if value > best_value:
#                 best_value = value
#                 best_index = point

#         # Return the optimized value
#         return best_value

def __call__(self, func, initial_individual, budget, max_iter):
    """
    Optimize the black box function using the BlackBoxOptimizer.

    Args:
        func (callable): The black box function to optimize.
        initial_individual (list): The initial individual for the optimization process.
        budget (int): The maximum number of function evaluations allowed.
        max_iter (int): The maximum number of iterations for the optimization process.

    Returns:
        float: The optimized value of the function.
    """
    # Initialize the best value and its corresponding index
    best_value = float('-inf')
    best_index = -1

    # Initialize the current individual
    current_individual = initial_individual

    # Initialize the population of individuals
    population = deque([current_individual])

    # Perform the specified number of iterations
    for _ in range(max_iter):
        # Evaluate the function at the current individual
        value = func(current_individual)

        # If the current value is better than the best value found so far,
        # update the best value and its corresponding index
        if value > best_value:
            best_value = value
            best_index = current_individual
        else:
            # If the current value is not better than the best value found so far,
            # replace the current individual with the best individual
            best_individual = best_index
            best_value = best_value
            best_index = np.random.randint(0, self.dim)

        # Add the best individual to the population
        population.append(best_individual)

        # If the maximum number of iterations is reached,
        # replace the current individual with the best individual
        if len(population) > self.budget:
            current_individual = population.popleft()

    # Return the optimized value
    return best_value

# Test the algorithm
def sphere(func):
    """
    The black box function to optimize.
    """
    return np.sum([x**2 for x in np.random.uniform(-5.0, 5.0)])

# Initialize the BlackBoxOptimizer with a budget of 1000 and a dimensionality of 5
optimizer = BlackBoxOptimizer(1000, 5)

# Optimize the sphere function using the BlackBoxOptimizer
optimized_value = optimizer(__call__, initial_individual=None)

# Print the optimized value
print(f"Optimized value: {optimized_value}")

# Test the algorithm with a different function
def boxcar(func):
    """
    The black box function to optimize.
    """
    return np.sum([x for i, x in enumerate(np.linspace(0, 10, 100)) for j, y in enumerate(np.linspace(0, 10, 100)) if i < 50 and j < 50])

# Initialize the BlackBoxOptimizer with a budget of 1000 and a dimensionality of 5
optimizer = BlackBoxOptimizer(1000, 5)

# Optimize the boxcar function using the BlackBoxOptimizer
optimized_value = optimizer(__call__, initial_individual=None)

# Print the optimized value
print(f"Optimized value: {optimized_value}")