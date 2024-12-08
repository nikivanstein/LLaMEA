import numpy as np
from scipy.optimize import minimize
from typing import Dict, Any
import random

class BlackBoxOptimizer:
    def __init__(self, budget: int, dim: int, func: Dict[str, float]) -> None:
        """
        Initialize the BlackBoxOptimizer with a given budget, dimension, and a black box function.

        Args:
        budget (int): The maximum number of function evaluations allowed.
        dim (int): The dimensionality of the search space.
        func (Dict[str, float]): A dictionary representing the black box function, where keys are the variable names and values are the function values.
        """
        self.budget = budget
        self.dim = dim
        self.func = func

    def __call__(self, func: Dict[str, float]) -> Dict[str, float]:
        """
        Optimize the black box function using an Evolutionary Algorithm.

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

        # Initialize the population with random individuals
        population = [x for _ in range(100)]

        # Evaluate the fitness of each individual and select the fittest ones
        for _ in range(self.budget):
            fitness = [objective(individual) for individual in population]
            selected_indices = np.argsort(fitness)[-self.budget:]
            selected_individuals = [population[i] for i in selected_indices]

            # Mutate the selected individuals with a probability of 0.45
            mutated_individuals = [random.choice(selected_individuals) for _ in range(100)]
            mutated_individuals = [x for x in mutated_individuals if random.random() < 0.45]

            # Replace the least fit individuals with the mutated ones
            population = [mutated_individuals[i] if fitness[i] < fitness[min(fitness)] else population[i] for i in range(100)]

        # Return the fittest individual
        return {k: -v for k, v in population[0].items()}

# Description: Evolutionary Algorithm for Optimization of Black Box Functions
# Code: 