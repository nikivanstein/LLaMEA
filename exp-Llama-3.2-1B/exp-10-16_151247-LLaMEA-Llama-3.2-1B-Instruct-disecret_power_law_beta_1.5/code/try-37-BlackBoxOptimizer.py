import numpy as np
import random

class BlackBoxOptimizer:
    def __init__(self, budget, dim):
        """
        Initialize the BlackBoxOptimizer with a given budget and dimensionality.

        Args:
        budget (int): The maximum number of function evaluations allowed.
        dim (int): The dimensionality of the search space.
        """
        self.budget = budget
        self.dim = dim
        self.search_space = np.random.uniform(-5.0, 5.0, (dim,))

    def __call__(self, func):
        """
        Optimize a black box function using the BlackBoxOptimizer.

        Args:
        func (function): The black box function to optimize.

        Returns:
        float: The optimized value of the function.
        """
        # Create a copy of the search space to avoid modifying the original
        search_space = self.search_space.copy()

        # Perform the specified number of function evaluations
        for _ in range(self.budget):
            # Evaluate the function at the current search space point
            func_value = func(search_space)

            # Update the search space point to be the next possible point
            # with adaptive mutation based on the fitness value
            new_individual = self.evaluate_fitness(search_space)
            mutation_rate = 0.1  # adjust this rate to balance exploration and exploitation
            new_individual = np.clip(new_individual, self.search_space[:, 0] - 2 * self.search_space[:, 0], self.search_space[:, 0] + 2 * self.search_space[:, 0])
            new_individual = np.clip(new_individual, self.search_space[:, 1] - 2 * self.search_space[:, 1], self.search_space[:, 1] + 2 * self.search_space[:, 1])
            new_individual = np.clip(new_individual, self.search_space[:, 2] - 2 * self.search_space[:, 2], self.search_space[:, 2] + 2 * self.search_space[:, 2])
            new_individual = np.clip(new_individual, self.search_space[:, 3] - 2 * self.search_space[:, 3], self.search_space[:, 3] + 2 * self.search_space[:, 3])

            # Apply random mutation to the new individual
            if random.random() < mutation_rate:
                new_individual = np.random.uniform(self.search_space[:, 0], self.search_space[:, 1], size=dim)
                new_individual = np.clip(new_individual, self.search_space[:, 2], self.search_space[:, 3])

            # Update the search space point to be the new individual
            search_space = new_individual

        # Return the optimized function value
        return func_value

    def evaluate_fitness(self, individual):
        """
        Evaluate the fitness of an individual using a given function.

        Args:
        individual (numpy array): The individual to evaluate.

        Returns:
        float: The fitness value of the individual.
        """
        func_value = 0
        for i in range(self.dim):
            func_value += individual[i] ** 2
        return func_value