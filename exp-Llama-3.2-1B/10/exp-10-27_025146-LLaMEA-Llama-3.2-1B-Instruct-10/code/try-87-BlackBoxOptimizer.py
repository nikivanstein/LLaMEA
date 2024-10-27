import random
import numpy as np

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

class NovelMetaheuristicOptimizer:
    def __init__(self, budget, dim):
        """
        Initialize the NovelMetaheuristicOptimizer with a budget and dimensionality.

        Args:
            budget (int): The maximum number of function evaluations allowed.
            dim (int): The dimensionality of the search space.
        """
        self.budget = budget
        self.dim = dim
        self.search_space = np.linspace(-5.0, 5.0, dim)

    def __call__(self, func, mutation_rate, crossover_rate):
        """
        Optimize the black box function using the NovelMetaheuristicOptimizer.

        Args:
            func (callable): The black box function to optimize.
            mutation_rate (float): The probability of mutation.
            crossover_rate (float): The probability of crossover.

        Returns:
            list: A list of optimized values.
        """
        # Initialize the best values and their corresponding indices
        best_values = []
        best_indices = []

        # Initialize the population with random points in the search space
        population = np.random.uniform(self.search_space, size=(self.budget, self.dim))

        # Perform the specified number of function evaluations
        for i in range(self.budget):
            # Generate a random point in the search space
            point = population[i]

            # Evaluate the function at the current point
            value = func(point)

            # If the current value is better than the best value found so far,
            # update the best value and its corresponding index
            if value > np.max(best_values):
                best_values.append(value)
                best_indices.append(i)

            # If mutation is applied, generate a new point in the search space
            if random.random() < mutation_rate:
                point = self.search_space[np.random.randint(0, self.dim)]

            # If crossover is applied, generate two new points in the search space
            if random.random() < crossover_rate:
                parent1 = population[np.random.randint(0, self.budget)]
                parent2 = population[np.random.randint(0, self.budget)]
                point = (parent1 + parent2) / 2

            # Evaluate the function at the new point
            value = func(point)

            # If the new value is better than the best value found so far,
            # update the best value and its corresponding index
            if value > np.max(best_values):
                best_values.append(value)
                best_indices.append(i)

        # Return the optimized values
        return best_values

# Example usage:
# optimizer = NovelMetaheuristicOptimizer(100, 5)
# func = lambda x: x**2
# optimized_values = optimizer(func, 0.1, 0.5)
# print(optimized_values)