import numpy as np
from scipy.optimize import minimize
from typing import Dict, Any
import random

class BlackBoxOptimizer:
    def __init__(self, budget: int, dim: int, func: Dict[str, float], mutation_prob: float) -> None:
        """
        Initialize the BlackBoxOptimizer with a given budget, dimension, and a black box function.

        Args:
        budget (int): The maximum number of function evaluations allowed.
        dim (int): The dimensionality of the search space.
        func (Dict[str, float]): A dictionary representing the black box function, where keys are the variable names and values are the function values.
        mutation_prob (float): The probability of mutating an individual.
        """
        self.budget = budget
        self.dim = dim
        self.func = func
        self.mutation_prob = mutation_prob

    def __call__(self, func: Dict[str, float]) -> Dict[str, float]:
        """
        Optimize the black box function using a novel heuristic algorithm.

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

        # Evaluate the optimized function values
        optimized_func_values = {k: -v for k, v in result.x.items()}

        # Evaluate the function values with the mutation strategy
        for _ in range(self.budget):
            new_individual = self.evaluate_fitness(x)
            mutation_prob = random.random()
            if mutation_prob < self.mutation_prob:
                new_individual = self.mutate(x)

            # Evaluate the mutated individual
            mutated_func_values = {k: -v for k, v in new_individual.items()}

            # Update the optimized function values
            optimized_func_values.update(mutated_func_values)

        return optimized_func_values

    def mutate(self, x: np.ndarray) -> np.ndarray:
        """
        Mutate an individual with a probability of mutation.

        Args:
        x (np.ndarray): The individual to mutate.

        Returns:
        np.ndarray: The mutated individual.
        """
        # Select a random index to mutate
        idx = random.randint(0, self.dim - 1)

        # Mutate the individual
        mutated_x = x.copy()
        mutated_x[idx] += random.uniform(-1.0, 1.0)

        return mutated_x