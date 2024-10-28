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
        Optimize the black box function using an adaptive heuristic algorithm.

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

        # Initialize the population size and the number of individuals to evolve
        population_size = 100
        num_individuals_to_evolve = int(self.budget / 10)

        # Initialize the population with random individuals
        population = np.random.uniform(low=-10, high=10, size=(population_size, self.dim))

        # Evolve the population for the specified number of individuals
        for _ in range(num_individuals_to_evolve):
            # Select the fittest individuals to reproduce
            fittest_individuals = population[np.argsort(-self.func.values(population[:, :self.dim]))[:num_individuals_to_evolve]]

            # Generate new individuals by perturbing the fittest individuals
            new_individuals = fittest_individuals.copy()
            for _ in range(self.dim):
                new_individuals[:, np.random.randint(0, self.dim)] += np.random.uniform(-1, 1)

            # Replace the least fit individuals with the new ones
            population = np.vstack((population[:num_individuals_to_evolve], new_individuals))

            # Update the bounds for the search space
            bounds = [(-5.0, 5.0) for _ in range(self.dim)]

            # Evaluate the fitness of the new population
            fitness_values = self.evaluate_fitness(population)

            # Update the population with the fittest individuals
            population = population[fitness_values.argsort()]

        # Evaluate the fitness of the final population
        fitness_values = self.evaluate_fitness(population)

        # Return the optimized function values
        return {k: -v for k, v in fitness_values.items()}

    def evaluate_fitness(self, population: np.ndarray) -> np.ndarray:
        """
        Evaluate the fitness of a given population.

        Args:
        population (np.ndarray): The population to evaluate.

        Returns:
        np.ndarray: The fitness values of the population.
        """
        # Define the objective function to minimize (negative of the original function)
        def objective(x: np.ndarray) -> float:
            return -np.sum(self.func.values(x))

        # Evaluate the fitness of the population
        fitness_values = np.sum(self.func.values(population), axis=1)

        return fitness_values

# Description: Adaptive Black Box Optimization Algorithm
# Code: 