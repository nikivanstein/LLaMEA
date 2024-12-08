# Description: A novel metaheuristic algorithm that uses a random search with bounds to optimize black box functions.
# Code: 
# ```python
import numpy as np
from scipy.optimize import minimize
from typing import Dict, Any

class BBOBMetaheuristic:
    """
    A metaheuristic algorithm for solving black box optimization problems.
    """

    def __init__(self, budget: int, dim: int):
        """
        Initialize the algorithm with a given budget and dimensionality.

        Args:
        budget (int): The maximum number of function evaluations allowed.
        dim (int): The dimensionality of the search space.
        """
        self.budget = budget
        self.dim = dim

    def __call__(self, func: Any, bounds: Dict[str, float] = None) -> Any:
        """
        Optimize the given black box function using the provided bounds.

        Args:
        func (Any): The black box function to optimize.
        bounds (Dict[str, float], optional): The bounds for the search space. Defaults to None.

        Returns:
        Any: The optimized function value.
        """
        # Create a copy of the function to avoid modifying the original function
        func_copy = func.copy()

        # Initialize the minimum and maximum function values
        min_func = np.inf
        max_func = -np.inf

        # Iterate over the budget iterations
        for _ in range(self.budget):
            # Generate random function values within the bounds
            random_func = np.random.uniform(-bounds["lower_bound"], bounds["upper_bound"], (self.dim,))

            # Evaluate the function using the random values
            func_value = func_copy(random_func)

            # Update the minimum and maximum function values
            min_func = min(min_func, func_value)
            max_func = max(max_func, func_value)

        # Return the optimized function value
        return min_func


# One-line description: A novel metaheuristic algorithm that uses a random search with bounds to optimize black box functions.
# Code: 
# ```python
def bbobmetaheuristic(budget: int, dim: int) -> float:
    """
    Optimize the given black box function using a random search with bounds.

    Args:
    budget (int): The maximum number of function evaluations allowed.
    dim (int): The dimensionality of the search space.

    Returns:
    float: The optimized function value.
    """
    # Initialize the population with random solutions
    population = np.random.uniform(-5, 5, (dim,))

    # Evolve the population for the specified budget
    for _ in range(budget):
        # Select the fittest individual
        fittest_individual = population[np.argmax(population[:, 0]), np.argmax(population[:, 1])]

        # Mutate the fittest individual
        mutated_individual = np.random.uniform(-5, 5, (dim,))
        mutated_individual[np.argmax(population[:, 0]), np.argmax(population[:, 1])] += np.random.uniform(-1, 1, (dim,))

        # Evaluate the mutated individual
        func_value = fittest_individual @ mutated_individual

        # Update the population
        population = np.concatenate((population, [fittest_individual, mutated_individual]), axis=0)

        # Check if the mutated individual is better than the current fittest individual
        if func_value < population[np.argmax(population[:, 0]), np.argmax(population[:, 1])]:
            population[np.argmax(population[:, 0]), np.argmax(population[:, 1])] = func_value

    # Return the fittest individual
    return population[np.argmax(population[:, 0]), np.argmax(population[:, 1])]


# Test the algorithm
budget = 100
dim = 10
best_solution = bbobmetaheuristic(budget, dim)
best_function_value = bbobmetaheuristic(budget, dim)  # Use the same function value as the initial solution
print(f"Best solution: {best_solution}, Best function value: {best_function_value}")