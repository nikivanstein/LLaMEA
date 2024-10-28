import numpy as np
from scipy.optimize import minimize
from typing import Dict, Any

class AdaptiveBlackBoxOptimizer:
    def __init__(self, budget: int, dim: int, func: Dict[str, float], alpha: float = 0.45, beta: float = 0.5) -> None:
        """
        Initialize the AdaptiveBlackBoxOptimizer with a given budget, dimension, and a black box function.

        Args:
        budget (int): The maximum number of function evaluations allowed.
        dim (int): The dimensionality of the search space.
        func (Dict[str, float]): A dictionary representing the black box function, where keys are the variable names and values are the function values.
        alpha (float, optional): The probability of changing the individual strategy. Defaults to 0.45.
        beta (float, optional): The probability of changing the bounds. Defaults to 0.5.
        """
        self.budget = budget
        self.dim = dim
        self.func = func
        self.alpha = alpha
        self.beta = beta
        self.new_individuals = []

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

        # Evaluate the new individual
        new_individual = self.evaluate_fitness(result.x)

        # Update the new individual based on the adaptive strategy
        if np.random.rand() < self.alpha:
            # Change the bounds with a probability of beta
            bounds = [(-5.0, 5.0) if np.random.rand() < beta else [5.0, -5.0] for _ in range(self.dim)]
        else:
            # Keep the current bounds
            pass

        # Add the new individual to the list of new individuals
        self.new_individuals.append(new_individual)

        # Return the optimized function values
        return {k: -v for k, v in result.x.items()}

    def evaluate_fitness(self, individual: np.ndarray) -> Dict[str, float]:
        """
        Evaluate the fitness of an individual.

        Args:
        individual (np.ndarray): The individual to evaluate.

        Returns:
        Dict[str, float]: The fitness of the individual.
        """
        # Evaluate the fitness of the individual using the original function
        fitness = np.sum(self.func.values(individual))

        # Update the individual with the new fitness
        individual = self.func[individual]

        # Return the new fitness
        return {k: -v for k, v in individual.items()}