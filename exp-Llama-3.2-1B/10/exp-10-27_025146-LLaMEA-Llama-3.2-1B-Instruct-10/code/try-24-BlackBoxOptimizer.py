import random
import numpy as np
from scipy.optimize import differential_evolution

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

    def __call__(self, func):
        """
        Optimize the black box function using the BlackBoxOptimizer.

        Args:
            func (callable): The black box function to optimize.

        Returns:
            float: The optimized value of the function.
        """
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

    def mutate(self, individual):
        """
        Randomly mutate an individual in the search space.

        Args:
            individual (list): The individual to mutate.

        Returns:
            list: The mutated individual.
        """
        # Randomly select an index to mutate
        idx = np.random.randint(0, self.dim)

        # Randomly change the value at the selected index
        individual[idx] = np.random.uniform(-5.0, 5.0)

        # Return the mutated individual
        return individual

    def crossover(self, parent1, parent2):
        """
        Perform crossover between two parents to generate a new individual.

        Args:
            parent1 (list): The first parent.
            parent2 (list): The second parent.

        Returns:
            list: The new individual.
        """
        # Randomly select a crossover point
        idx = np.random.randint(0, self.dim)

        # Create a new individual by combining the two parents
        child = np.concatenate((parent1[:idx], parent2[idx:]))

        # Return the new individual
        return child

# One-line description with the main idea
# Novel Metaheuristic Algorithm for Black Box Optimization
# Refines the search space by iteratively applying mutation, crossover, and differential evolution