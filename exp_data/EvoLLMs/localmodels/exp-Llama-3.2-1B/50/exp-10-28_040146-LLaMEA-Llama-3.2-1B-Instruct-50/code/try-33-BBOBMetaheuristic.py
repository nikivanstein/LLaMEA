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

    def __call__(self, func: Any, bounds: Dict[str, float] = None, mutation_rate: float = 0.01) -> Any:
        """
        Optimize the given black box function using the provided bounds and mutation rate.

        Args:
        func (Any): The black box function to optimize.
        bounds (Dict[str, float], optional): The bounds for the search space. Defaults to None.
        mutation_rate (float, optional): The rate at which to mutate the individual. Defaults to 0.01.

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

        # If mutation rate is greater than 0, mutate the individual
        if mutation_rate > 0:
            # Generate a random mutation
            mutation = np.random.uniform(-1, 1, (self.dim,))

            # Mutate the individual
            updated_individual = func_copy(np.append(random_func, mutation))

            # Evaluate the mutated individual
            updated_func_value = minimize(lambda x: x[0]**2 + x[1]**2, updated_individual, method="SLSQP", bounds=[(-5, 5), (-5, 5)])

            # Update the minimum and maximum function values
            min_func = min(min_func, updated_func_value.fun)
            max_func = max(max_func, updated_func_value.fun)

        # Return the optimized function value
        return min_func


# One-line description: A novel metaheuristic algorithm that uses a random search with bounds to optimize black box functions.
# Code: 
# ```python
# import numpy as np
# import scipy.optimize as optimize
#
# def bbobmetaheuristic(budget: int, dim: int) -> float:
#     return optimize.minimize(lambda x: x[0]**2 + x[1]**2, [1, 1], method="SLSQP", bounds=[(-5, 5), (-5, 5)], mutation_rate=0.01, max_iter=100)