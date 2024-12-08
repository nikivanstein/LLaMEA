import numpy as np
import random
from typing import Dict, Any, List

class BlackBoxOptimizer:
    def __init__(self, budget: int, dim: int, func: Dict[str, float], mutation_rate: float) -> None:
        """
        Initialize the BlackBoxOptimizer with a given budget, dimension, a black box function, and a mutation rate.

        Args:
        budget (int): The maximum number of function evaluations allowed.
        dim (int): The dimensionality of the search space.
        func (Dict[str, float]): A dictionary representing the black box function, where keys are the variable names and values are the function values.
        mutation_rate (float): The probability of introducing a mutation in the current individual.
        """
        self.budget = budget
        self.dim = dim
        self.func = func
        self.mutation_rate = mutation_rate
        self.current_individual = None
        self.population_size = 100
        self.population = self.initialize_population()

    def initialize_population(self) -> List[Dict[str, float]]:
        """
        Initialize the population with random individuals.

        Returns:
        List[Dict[str, float]]: The initialized population.
        """
        return [[random.uniform(-5.0, 5.0) for _ in range(self.dim)] for _ in range(self.population_size)]

    def evaluate_fitness(self, individual: Dict[str, float]) -> Dict[str, float]:
        """
        Evaluate the fitness of an individual.

        Args:
        individual (Dict[str, float]): The individual to evaluate.

        Returns:
        Dict[str, float]: The fitness of the individual.
        """
        func_values = self.func.values(individual)
        return {k: v for k, v in zip(list(func_values.keys()), func_values)}

    def __call__(self, func: Dict[str, float]) -> Dict[str, float]:
        """
        Optimize the black box function using a novel heuristic algorithm.

        Args:
        func (Dict[str, float]): A dictionary representing the black box function, where keys are the variable names and values are the function values.

        Returns:
        Dict[str, float]: The optimized function values.
        """
        # Initialize the search space with random values
        self.current_individual = random.choice(self.population)
        self.population = self.population[:]

        # Define the objective function to minimize (negative of the original function)
        def objective(individual: Dict[str, float]) -> float:
            return -np.sum(self.func.values(individual))

        # Define the bounds for the search space
        bounds = [(-5.0, 5.0) for _ in range(self.dim)]

        # Use the minimize function to optimize the black box function
        result = minimize(objective, self.current_individual, method="SLSQP", bounds=bounds, constraints={"type": "eq", "fun": lambda x: 0}, x0=self.current_individual)

        # Update the current individual with the optimized values
        self.current_individual = result.x

        # If the budget is exhausted, return the current individual
        if len(self.population) < self.budget:
            return self.evaluate_fitness(self.current_individual)

        # Refine the strategy by changing the mutation rate
        if random.random() < 0.5:
            self.mutation_rate *= 0.9

        # Return the optimized function values
        return {k: -v for k, v in result.x.items()}

# Description: BlackBoxOptimizer: A novel metaheuristic algorithm for solving black box optimization problems.
# Code: 