import random
import numpy as np
from scipy.optimize import minimize

class BlackBoxOptimizer:
    def __init__(self, budget, dim):
        """
        Initialize the BlackBoxOptimizer with a budget and dimensionality.

        Args:
            budget (int): The maximum number of function evaluations allowed.
            dim (int): The dimensionality of the search space.
        """
        self.budget = budget
        self.dim = dim
        self.search_space = np.linspace(-5.0, 5.0, dim)

    def __call__(self, func, initial_individual=None):
        """
        Optimize the black box function using the BlackBoxOptimizer.

        Args:
            func (callable): The black box function to optimize.
            initial_individual (list[float], optional): The initial individual. Defaults to None.

        Returns:
            float: The optimized value of the function.
        """
        # If no initial individual is provided, use the current best value
        if initial_individual is None:
            initial_individual = self.evaluate_fitness(self.evaluate_individual())

        # Initialize the best value and its corresponding index
        best_value = float('-inf')
        best_index = -1

        # Perform the specified number of function evaluations
        for _ in range(self.budget):
            # Generate a random point in the search space
            point = self.search_space[np.random.randint(0, self.dim)]

            # Evaluate the function at the current point
            value = func(point)

            # If the current value is better than the best value found so far,
            # update the best value and its corresponding index
            if value > best_value:
                best_value = value
                best_index = point

        # Return the optimized value
        return best_value

    def evaluate_individual(self, individual):
        """
        Evaluate the individual using the BlackBoxOptimizer.

        Args:
            individual (list[float]): The individual to evaluate.

        Returns:
            float: The fitness of the individual.
        """
        # Evaluate the individual using the BlackBoxOptimizer
        value = self.evaluate_fitness(individual)
        return value

    def evaluate_fitness(self, individual):
        """
        Evaluate the fitness of the individual.

        Args:
            individual (list[float]): The individual to evaluate.

        Returns:
            float: The fitness of the individual.
        """
        # Evaluate the fitness using the BlackBoxOptimizer
        return self.func(individual)

    def func(self, individual):
        """
        Evaluate the fitness of the individual using the specified function.

        Args:
            individual (list[float]): The individual to evaluate.

        Returns:
            float: The fitness of the individual.
        """
        # Use the specified function to evaluate the fitness
        return minimize(self.func, individual, method="SLSQP", bounds=self.search_space)