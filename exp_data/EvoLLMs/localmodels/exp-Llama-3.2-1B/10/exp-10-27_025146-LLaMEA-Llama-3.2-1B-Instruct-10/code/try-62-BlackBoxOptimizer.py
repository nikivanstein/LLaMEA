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

    def __call__(self, func):
        """
        Optimize the black box function using the NovelMetaheuristicOptimizer.

        Args:
            func (callable): The black box function to optimize.

        Returns:
            float: The optimized value of the function.
        """
        # Initialize the best value and its corresponding index
        best_value = float('-inf')
        best_index = -1

        # Initialize the mutation rate
        mutation_rate = 0.01

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

            # If the best value found so far is not better than the current value,
            # mutate the current point
            if best_value == float('-inf'):
                point = self.mutate(point, mutation_rate)

            # Update the current point in the search space
            self.search_space[best_index] = point

        # Return the optimized value
        return best_value

def mutate(individual, mutation_rate):
    """
    Mutate an individual in the search space.

    Args:
        individual (List[float]): The individual to mutate.
        mutation_rate (float): The mutation rate.

    Returns:
        List[float]: The mutated individual.
    """
    # Generate a random mutation
    mutation = random.uniform(-mutation_rate, mutation_rate)

    # Mutate the individual
    mutated_individual = individual.copy()
    mutated_individual[np.random.randint(0, len(individual))] += mutation

    return mutated_individual

# Description: Novel Metaheuristic Algorithm for Black Box Optimization
# Code: 