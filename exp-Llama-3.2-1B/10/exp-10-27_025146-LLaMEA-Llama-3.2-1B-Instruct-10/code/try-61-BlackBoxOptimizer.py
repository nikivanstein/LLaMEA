# Description: Novel Metaheuristic Algorithm for Black Box Optimization
# Code: 
# ```python
import random
import numpy as np
from scipy.optimize import minimize
import copy

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
            point = copy.deepcopy(self.search_space[np.random.randint(0, self.dim)])

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
        # Generate a random mutation
        mutation = random.uniform(-1.0, 1.0)

        # Mutate the individual
        mutated_individual = copy.deepcopy(individual)
        mutated_individual[0] += mutation

        return mutated_individual

    def crossover(self, individual1, individual2):
        """
        Perform crossover between two individuals in the search space.

        Args:
            individual1 (list): The first individual.
            individual2 (list): The second individual.

        Returns:
            list: The resulting individual after crossover.
        """
        # Select a random crossover point
        crossover_point = np.random.randint(0, self.dim)

        # Perform crossover
        child1 = copy.deepcopy(individual1)
        child2 = copy.deepcopy(individual2)

        child1[crossover_point:] = child2[:crossover_point]

        return child1, child2

# Description: Novel Metaheuristic Algorithm for Black Box Optimization
# Code: 
# ```python
black_box_optimizer = BlackBoxOptimizer(budget=100, dim=5)