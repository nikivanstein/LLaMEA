import numpy as np
from scipy.optimize import minimize
from typing import Dict, Any

class AdaptiveBlackBoxOptimizer:
    def __init__(self, budget: int, dim: int, func: Dict[str, float], mutation_rate: float, selection_rate: float) -> None:
        """
        Initialize the AdaptiveBlackBoxOptimizer with a given budget, dimension, a black box function, mutation rate, and selection rate.

        Args:
        budget (int): The maximum number of function evaluations allowed.
        dim (int): The dimensionality of the search space.
        func (Dict[str, float]): A dictionary representing the black box function, where keys are the variable names and values are the function values.
        mutation_rate (float): The probability of mutation in each generation.
        selection_rate (float): The probability of selection in each generation.
        """
        self.budget = budget
        self.dim = dim
        self.func = func
        self.mutation_rate = mutation_rate
        self.selection_rate = selection_rate
        self.population_size = 100
        self.population = self.initialize_population()

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

        # Update the population with the optimized function values
        self.population = self.update_population(result.x, self.budget, self.mutation_rate, self.selection_rate)

        # Return the optimized function values
        return {k: -v for k, v in self.population[0].items()}

    def initialize_population(self) -> Dict[str, float]:
        """
        Initialize the population with random function values.

        Returns:
        Dict[str, float]: A dictionary representing the population, where keys are the variable names and values are the function values.
        """
        return {k: np.random.uniform(-5.0, 5.0) for k in self.func.keys()}

    def update_population(self, new_individual: Dict[str, float], budget: int, mutation_rate: float, selection_rate: float) -> Dict[str, float]:
        """
        Update the population with new individuals.

        Args:
        new_individual (Dict[str, float]): A dictionary representing the new individual, where keys are the variable names and values are the function values.
        budget (int): The remaining budget for function evaluations.
        mutation_rate (float): The probability of mutation in each individual.
        selection_rate (float): The probability of selection in each generation.

        Returns:
        Dict[str, float]: The updated population.
        """
        # Filter out individuals that exceed the budget
        filtered_population = {k: v for k, v in new_individual.items() if v <= budget}

        # Select the best individuals according to the selection rate
        selected_population = {k: v for k, v in filtered_population.items() if np.random.rand() < self.selection_rate}

        # Create new individuals by mutating the selected individuals
        new_population = {}
        for k, v in selected_population.items():
            # Create a copy of the individual
            new_individual = new_population[k] = {k: v}

            # Mutate the individual with the specified mutation rate
            for i in range(self.dim):
                mutation_rate = self.mutation_rate
                if np.random.rand() < mutation_rate:
                    new_individual[k][i] += np.random.uniform(-1, 1)

            # Update the population with the new individual
            new_population[k] = new_individual

        return new_population