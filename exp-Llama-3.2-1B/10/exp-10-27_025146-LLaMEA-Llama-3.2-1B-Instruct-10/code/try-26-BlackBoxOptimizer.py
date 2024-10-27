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

        # Update the mutation strategy based on the best value found so far
        if best_value!= float('-inf'):
            # Select a random mutation point
            mutation_point = np.random.choice(self.search_space)

            # Perform a small mutation on the selected point
            mutated_point = mutation_point + random.uniform(-0.1, 0.1)

            # Evaluate the function at the mutated point
            mutated_value = func(mutated_point)

            # If the mutated value is better than the best value found so far,
            # update the best value and its corresponding index
            if mutated_value > best_value:
                best_value = mutated_value
                best_index = mutated_point

        # Return the optimized value
        return best_value

# Description: Novel Metaheuristic Algorithm for Black Box Optimization using Adaptive Mutation Strategy
# Code: 