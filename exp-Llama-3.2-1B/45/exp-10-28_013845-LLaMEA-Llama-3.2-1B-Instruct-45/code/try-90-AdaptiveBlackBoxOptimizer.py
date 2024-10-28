import numpy as np
from scipy.optimize import minimize
from typing import Dict, Any

class AdaptiveBlackBoxOptimizer:
    def __init__(self, budget: int, dim: int, func: Dict[str, float], p1: float, p2: float) -> None:
        """
        Initialize the AdaptiveBlackBoxOptimizer with a given budget, dimension, and a black box function.

        Args:
        budget (int): The maximum number of function evaluations allowed.
        dim (int): The dimensionality of the search space.
        func (Dict[str, float]): A dictionary representing the black box function, where keys are the variable names and values are the function values.
        p1 (float): The proportion of the population to adaptively change.
        p2 (float): The proportion of the population to use the fixed strategy.
        """
        self.budget = budget
        self.dim = dim
        self.func = func
        self.p1 = p1
        self.p2 = p2
        self.population = [np.random.uniform(-5.0, 5.0, self.dim) for _ in range(100)]

    def __call__(self, func: Dict[str, float]) -> Dict[str, float]:
        """
        Optimize the black box function using an adaptive strategy.

        Args:
        func (Dict[str, float]): A dictionary representing the black box function, where keys are the variable names and values are the function values.

        Returns:
        Dict[str, float]: The optimized function values.
        """
        # Initialize the search space with random values
        x = np.random.uniform(-5.0, 5.0, self.dim)

        # Define the objective function to minimize (negative of the original function)
        def objective(x: np.ndarray) -> float:
            return -np.sum(self.func.values(x))

        # Define the bounds for the search space
        bounds = [(-5.0, 5.0) for _ in range(self.dim)]

        # Use the minimize function to optimize the black box function
        result = minimize(objective, x, method="SLSQP", bounds=bounds, constraints={"type": "eq", "fun": lambda x: 0}, x0=x)

        # Update the population based on the adaptive strategy
        if np.random.rand() < self.p1:
            self.population = [np.random.uniform(-5.0, 5.0, self.dim) for _ in range(100)]
        else:
            self.population = [np.copy(self.population[-1])]

        # Update the fitness values
        self.population = [self.evaluate_fitness(individual, self.func) for individual in self.population]

        # Return the optimized function values
        return {k: -v for k, v in self.population[0].items()}

    def evaluate_fitness(self, individual: np.ndarray, func: Dict[str, float]) -> Dict[str, float]:
        """
        Evaluate the fitness of an individual.

        Args:
        individual (np.ndarray): The individual to evaluate.
        func (Dict[str, float]): The black box function.

        Returns:
        Dict[str, float]: The fitness value.
        """
        # Use the minimize function to optimize the black box function
        result = minimize(lambda x: -np.sum(func.values(x)), individual, method="SLSQP", bounds=[(-5.0, 5.0) for _ in range(self.dim)], constraints={"type": "eq", "fun": lambda x: 0})

        # Return the fitness value
        return {k: -v for k, v in result.x.items()}

# Description: Adaptive Black Box Optimizer (ABBO)
# Code: 
# ```python
# AdaptiveBlackBoxOptimizer: An adaptive metaheuristic algorithm for solving black box optimization problems.
# Code: 
# ```python