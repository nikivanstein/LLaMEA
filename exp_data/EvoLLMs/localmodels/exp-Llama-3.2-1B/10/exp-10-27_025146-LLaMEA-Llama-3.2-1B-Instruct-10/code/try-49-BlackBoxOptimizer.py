import random
import numpy as np
import copy
import math

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
        # Select a random point in the search space
        index = np.random.randint(0, self.dim)

        # Mutate the individual by swapping the point with a random point in the search space
        mutated_individual = copy.deepcopy(individual)
        mutated_individual[index], mutated_individual[index + 1] = mutated_individual[index + 1], mutated_individual[index]

        return mutated_individual

    def annealing(self, initial_value, cooling_rate):
        """
        Simulate the process of simulated annealing.

        Args:
            initial_value (float): The initial value of the function.
            cooling_rate (float): The cooling rate for the simulated annealing process.

        Returns:
            float: The optimized value of the function.
        """
        # Initialize the best value and its corresponding index
        best_value = float('-inf')
        best_index = -1

        # Initialize the current value and its corresponding index
        current_value = initial_value
        current_index = 0

        # Perform the specified number of function evaluations
        for _ in range(self.budget):
            # Generate a random point in the search space
            point = self.search_space[current_index]

            # Evaluate the function at the current point
            value = func(point)

            # If the current value is better than the best value found so far,
            # update the best value and its corresponding index
            if value > best_value:
                best_value = value
                best_index = current_index

            # If the current value is not better than the best value found so far,
            # and the current value is less than the best value found so far,
            # then accept the current value with a probability based on the cooling rate
            if value < best_value and math.exp((current_value - best_value) / 1000) > math.exp((best_value - current_value) / 1000):
                current_value = value
                current_index = 0

        # Return the optimized value
        return best_value