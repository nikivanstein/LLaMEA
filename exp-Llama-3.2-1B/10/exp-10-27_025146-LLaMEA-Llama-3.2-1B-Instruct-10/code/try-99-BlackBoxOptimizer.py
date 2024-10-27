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

    def __call__(self, func, mutation_prob=0.1, crossover_prob=0.1):
        """
        Optimize the black box function using the BlackBoxOptimizer.

        Args:
            func (callable): The black box function to optimize.
            mutation_prob (float, optional): The probability of mutation. Defaults to 0.1.
            crossover_prob (float, optional): The probability of crossover. Defaults to 0.1.

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

            # Perform mutation
            if random.random() < mutation_prob:
                # Generate a new point by adding a random value from the search space
                new_point = point + np.random.uniform(-5.0, 5.0, self.dim)
                # Evaluate the function at the new point
                new_value = func(new_point)

                # If the new value is better than the best value found so far,
                # update the best value and its corresponding index
                if new_value > best_value:
                    best_value = new_value
                    best_index = new_point

            # Perform crossover
            if random.random() < crossover_prob:
                # Select two random points from the search space
                parent1 = point
                parent2 = point

                # Crossover the two points to generate a new point
                child = (parent1 + parent2) / 2

                # Evaluate the function at the child point
                child_value = func(child)

                # If the child value is better than the best value found so far,
                # update the best value and its corresponding index
                if child_value > best_value:
                    best_value = child_value
                    best_index = child

        # Return the optimized value
        return best_value