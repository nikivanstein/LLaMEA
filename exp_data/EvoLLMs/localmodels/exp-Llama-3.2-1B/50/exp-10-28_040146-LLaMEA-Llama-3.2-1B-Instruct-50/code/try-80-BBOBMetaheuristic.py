# Description: A novel metaheuristic algorithm that uses a combination of bounds-based and adaptive search strategies to optimize black box functions.
# Code: 
# ```python
import numpy as np
import scipy.optimize as optimize
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

    def __call__(self, func: Any, bounds: Dict[str, float] = None, adaptive=True) -> Any:
        """
        Optimize the given black box function using the provided bounds and adaptive search strategy.

        Args:
        func (Any): The black box function to optimize.
        bounds (Dict[str, float], optional): The bounds for the search space. Defaults to None.
        adaptive (bool, optional): Whether to use an adaptive search strategy. Defaults to True.

        Returns:
        Any: The optimized function value.
        """
        # Create a copy of the function to avoid modifying the original function
        func_copy = func.copy()

        # Initialize the minimum and maximum function values
        min_func = np.inf
        max_func = -np.inf

        # Initialize the best individual and its fitness
        best_individual = None
        best_fitness = float('-inf')

        # Iterate over the budget iterations
        for _ in range(self.budget):
            # Generate random function values within the bounds
            random_func = np.random.uniform(-bounds["lower_bound"], bounds["upper_bound"], (self.dim,))

            # Evaluate the function using the random values
            func_value = func_copy(random_func)

            # Update the minimum and maximum function values
            min_func = min(min_func, func_value)
            max_func = max(max_func, func_value)

            # Evaluate the fitness of the current individual
            fitness = np.sqrt(np.sum((func_copy - random_func) ** 2))

            # Update the best individual and its fitness
            if fitness > best_fitness:
                best_individual = random_func
                best_fitness = fitness

            # Update the minimum and maximum function values
            if adaptive:
                # Use an adaptive search strategy to refine the bounds
                min_func = min(min_func, np.min([func_copy(random_func) for random_func in np.random.uniform(bounds["lower_bound"], bounds["upper_bound"], (self.dim,))]))
                max_func = max(max_func, np.max([func_copy(random_func) for random_func in np.random.uniform(bounds["lower_bound"], bounds["upper_bound"], (self.dim,))]))

        # Return the optimized function value
        return min_func


# Initialize the algorithm with a given budget and dimensionality
bboo_metaheuristic = BBOBMetaheuristic(100, 10)

# Evaluate the fitness of the initial individual
initial_fitness = bboo_metaheuristic(func, adaptive=False)
print(f"Initial fitness: {initial_fitness}")

# Optimize the function using the adaptive search strategy
optimized_fitness = bboo_metaheuristic(func, adaptive=True)
print(f"Optimized fitness: {optimized_fitness}")