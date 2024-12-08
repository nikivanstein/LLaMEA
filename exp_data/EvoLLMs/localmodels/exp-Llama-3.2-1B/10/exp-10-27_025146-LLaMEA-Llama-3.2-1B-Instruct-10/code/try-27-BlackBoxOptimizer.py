# Description: Novel Metaheuristic Algorithm for Black Box Optimization
# Code: 
# ```python
import random
import numpy as np
from scipy.optimize import minimize
from copy import deepcopy

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

    def __call__(self, func, initial_individual, logger):
        """
        Optimize the black box function using the BlackBoxOptimizer.

        Args:
            func (callable): The black box function to optimize.
            initial_individual (List[float]): The initial individual to use.
            logger (Logger): The logger to use for logging progress.

        Returns:
            float: The optimized value of the function.
        """
        # Perform the specified number of function evaluations
        for _ in range(self.budget):
            # Generate a random point in the search space
            point = initial_individual + np.random.normal(0, 1, self.dim)

            # Evaluate the function at the current point
            value = func(point)

            # If the current value is better than the best value found so far,
            # update the best value and its corresponding index
            if value > self.evaluate_fitness(point, logger):
                self.search_space = deepcopy(self.search_space)
                self.search_space = np.linspace(-5.0, 5.0, self.dim)

                # Update the individual with the new point
                new_individual = point
                updated_individual = deepcopy(initial_individual)
                updated_individual[0] = new_individual[0]
                new_individual[0] = updated_individual[0]

                updated_individual = self.evaluate_fitness(updated_individual, logger)

                self.search_space = deepcopy(self.search_space)
                self.search_space = np.linspace(-5.0, 5.0, self.dim)

                # Update the best value and its corresponding index
                best_value = value
                best_index = point

        # Return the optimized value
        return best_value

    def evaluate_fitness(self, individual, logger):
        """
        Evaluate the fitness of an individual.

        Args:
            individual (List[float]): The individual to evaluate.
            logger (Logger): The logger to use for logging progress.

        Returns:
            float: The fitness of the individual.
        """
        # Calculate the fitness of the individual
        fitness = func(individual, logger)
        logger.info(f'Fitness: {fitness}')

        # Return the fitness
        return fitness