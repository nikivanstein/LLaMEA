import numpy as np
from scipy.optimize import minimize

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


# Description: A novel metaheuristic algorithm that uses a combination of random search and adaptive bounds to optimize black box functions.
# Code: 
# ```python
def bbobmetaheuristic(budget: int, dim: int) -> float:
    """
    Optimize the given black box function using a random search with adaptive bounds.

    Args:
    budget (int): The maximum number of function evaluations allowed.
    dim (int): The dimensionality of the search space.

    Returns:
    float: The optimized function value.
    """
    # Initialize the population of random individuals
    population = np.random.uniform(-5.0, 5.0, (dim,))

    # Initialize the best individual and its fitness
    best_individual = population[0]
    best_fitness = bbobmetaheuristic.budget

    # Initialize the number of iterations
    iterations = 0

    # Iterate until the budget is reached
    while iterations < budget:
        # Evaluate the fitness of each individual
        fitness = bbobmetaheuristic(budget, dim)(best_individual, bounds=None)

        # Select the individual with the best fitness
        if fitness < best_fitness:
            best_individual = population[np.argmax(fitness)]
            best_fitness = fitness

        # Generate new individuals using the selected individual and adaptive bounds
        new_individuals = []
        for _ in range(dim):
            # Generate random function values within the adaptive bounds
            random_func = np.random.uniform(-bounds["lower_bound"], bounds["upper_bound"], (1,))

            # Evaluate the function using the random values
            func_value = bbobmetaheuristic(budget, 1)(random_func)

            # Update the new individual
            new_individual = np.random.uniform(-bounds["lower_bound"], bounds["upper_bound"], (1,))

            # Add the new individual to the list
            new_individuals.append(new_individual)

        # Replace the old population with the new population
        population = np.array(new_individuals)

        # Increment the number of iterations
        iterations += 1

    # Return the best individual and its fitness
    return bbobmetaheuristic(budget, dim)(best_individual, bounds=None)


# Example usage:
best_solution = bbobmetaheuristic(budget=100, dim=5)
print(f"Best solution: {best_solution}")
print(f"Best fitness: {bbobmetaheuristic(budget=100, dim=5)}")