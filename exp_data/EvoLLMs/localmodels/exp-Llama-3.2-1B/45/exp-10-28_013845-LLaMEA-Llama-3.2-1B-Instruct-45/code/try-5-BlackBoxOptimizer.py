import numpy as np
from scipy.optimize import minimize
from typing import Dict, Any

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
        Optimize the black box function using an adaptation-based strategy.

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

        # Apply adaptation to the optimized solution
        adapted_individual = self.adapt_solution(result.x, self.budget)

        # Return the adapted function values
        return {k: -v for k, v in adapted_individual.items()}

    def adapt_solution(self, solution: np.ndarray, budget: int) -> np.ndarray:
        """
        Adapt the optimized solution using a strategy based on the probability of convergence.

        Args:
        solution (np.ndarray): The optimized solution.
        budget (int): The remaining number of function evaluations allowed.

        Returns:
        np.ndarray: The adapted solution.
        """
        # Calculate the number of function evaluations required for convergence
        num_evaluations = np.ceil(budget / 0.45)

        # Initialize the adapted solution
        adapted_individual = np.copy(solution)

        # Iterate over the remaining evaluations
        for i in range(num_evaluations):
            # Generate a new individual using mutation and crossover
            new_individual = self.mutate_and_crossover(adapted_individual, self.dim)

            # Evaluate the new individual
            new_fitness = self.evaluate_fitness(new_individual)

            # If the new individual is better, update the adapted solution
            if new_fitness < adapted_individual[i, :]:
                adapted_individual[i, :] = new_individual

        return adapted_individual

    def mutate_and_crossover(self, individual: np.ndarray, dim: int) -> np.ndarray:
        """
        Mutate and crossover a single individual to generate a new individual.

        Args:
        individual (np.ndarray): The individual to mutate and crossover.
        dim (int): The dimensionality of the search space.

        Returns:
        np.ndarray: The new individual.
        """
        # Randomly select a mutation point
        mutation_point = np.random.randint(0, dim)

        # Mutate the individual at the selected point
        mutated_individual = individual.copy()
        mutated_individual[mutation_point] += np.random.uniform(-1, 1)

        # Crossover the mutated individual with another individual
        crossover_point1 = np.random.randint(0, dim)
        crossover_point2 = np.random.randint(0, dim)
        crossover_point = np.min([crossover_point1, crossover_point2])
        crossover_individual1 = individual.copy()
        crossover_individual1[crossover_point:]=mutated_individual[crossover_point:crossover_point+dim]
        crossover_individual2 = individual.copy()
        crossover_individual2[crossover_point:]=mutated_individual[crossover_point+1:2*dim]

        # Return the new individual
        return np.concatenate((crossover_individual1, crossover_individual2))

    def evaluate_fitness(self, individual: np.ndarray) -> float:
        """
        Evaluate the fitness of an individual using the objective function.

        Args:
        individual (np.ndarray): The individual to evaluate.

        Returns:
        float: The fitness of the individual.
        """
        return -np.sum(self.func.values(individual))