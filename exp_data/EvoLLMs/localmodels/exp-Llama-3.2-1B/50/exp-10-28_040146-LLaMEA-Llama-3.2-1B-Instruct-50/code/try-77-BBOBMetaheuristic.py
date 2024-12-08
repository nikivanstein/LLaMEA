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

    def select_new_individual(self, individual: Any, fitness: Any, logger: Any) -> Any:
        """
        Select a new individual based on the probability of 0.45.

        Args:
        individual (Any): The current individual.
        fitness (Any): The fitness of the individual.
        logger (Any): The logger to use for logging.

        Returns:
        Any: The selected individual.
        """
        # Select the individual with the highest fitness based on the probability
        selected_individual = individual[np.argmax(fitness)]
        # Refine the strategy by changing the individual lines
        selected_individual = np.random.choice(["line1", "line2"], p=[0.55, 0.45])
        return selected_individual

    def mutate(self, individual: Any, logger: Any) -> Any:
        """
        Mutate the current individual by changing a random gene.

        Args:
        individual (Any): The current individual.
        logger (Any): The logger to use for logging.

        Returns:
        Any: The mutated individual.
        """
        # Randomly select a gene to mutate
        gene_to_mutate = np.random.choice(list(individual.keys()))
        # Change the mutated gene
        mutated_individual = individual.copy()
        mutated_individual[gene_to_mutate] = np.random.uniform(-1, 1)
        return mutated_individual

# One-line description: A novel metaheuristic algorithm that uses a random search with bounds to optimize black box functions.
# Code: 
# ```python
# import numpy as np
# import scipy.optimize as optimize
# import random

class BBOBMetaheuristic:
    def __init__(self, budget: int, dim: int):
        self.budget = budget
        self.dim = dim

    def __call__(self, func: Any, bounds: Dict[str, float] = None) -> Any:
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

    def select_new_individual(self, individual: Any, fitness: Any, logger: Any) -> Any:
        # Select a new individual based on the probability of 0.45
        selected_individual = individual[np.argmax(fitness)]

        # Refine the strategy by changing the individual lines
        selected_individual = random.choice(["line1", "line2"], p=[0.55, 0.45])
        return selected_individual

    def mutate(self, individual: Any, logger: Any) -> Any:
        # Mutate the current individual by changing a random gene
        mutated_individual = individual.copy()
        mutated_individual[random.choice(list(individual.keys()))] = np.random.uniform(-1, 1)
        return mutated_individual