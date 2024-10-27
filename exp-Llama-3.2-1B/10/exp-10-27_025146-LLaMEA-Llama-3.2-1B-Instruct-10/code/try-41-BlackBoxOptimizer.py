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

        # Refine the search space using adaptive line search
        if best_value == float('-inf'):
            best_index = self.search_space[np.random.randint(0, self.dim)]

        # Perform mutation operations to refine the search space
        for _ in range(self.budget // 2):
            # Select a random individual from the current population
            individual = np.random.choice(self.population, 1)

            # Apply a mutation operator to the selected individual
            if random.random() < 0.1:
                individual = np.random.uniform(self.search_space)

            # Evaluate the mutated individual
            mutated_value = func(individual)

            # If the mutated value is better than the best value found so far,
            # update the best value and its corresponding index
            if mutated_value > best_value:
                best_value = mutated_value
                best_index = individual

        # Return the optimized value
        return best_value

# One-line description with the main idea
# Novel Metaheuristic Algorithm for Black Box Optimization
# Combines population-based approach with genetic algorithm for optimization