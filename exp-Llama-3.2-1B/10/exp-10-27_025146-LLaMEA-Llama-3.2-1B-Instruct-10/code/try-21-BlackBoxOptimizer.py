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

    def __call__(self, func, population_size=100, mutation_rate=0.01, n_iter=100):
        """
        Optimize the black box function using the BlackBoxOptimizer.

        Args:
            func (callable): The black box function to optimize.
            population_size (int, optional): The size of the population. Defaults to 100.
            mutation_rate (float, optional): The probability of mutation. Defaults to 0.01.
            n_iter (int, optional): The number of iterations. Defaults to 100.

        Returns:
            float: The optimized value of the function.
        """
        # Initialize the best value and its corresponding index
        best_value = float('-inf')
        best_index = -1

        # Perform the specified number of function evaluations
        for _ in range(min(self.budget, n_iter)):
            # Initialize the population with random points in the search space
            population = [self.search_space[np.random.randint(0, self.dim)] for _ in range(population_size)]

            # Evaluate the function at each point in the population
            for point in population:
                value = func(point)

                # If the current value is better than the best value found so far,
                # update the best value and its corresponding index
                if value > best_value:
                    best_value = value
                    best_index = point

            # Perform mutation on the best point
            if random.random() < mutation_rate:
                mutated_point = self.search_space[np.random.randint(0, self.dim)]
                mutated_point[best_index] = np.random.uniform(-5.0, 5.0)

            # Replace the worst point with the mutated point
            population[best_index] = mutated_point

            # Replace the current population with the new population
            population = population[:population_size]

        # Return the optimized value
        return best_value

# Description: Novel Metaheuristic Algorithm for Black Box Optimization
# Code: 