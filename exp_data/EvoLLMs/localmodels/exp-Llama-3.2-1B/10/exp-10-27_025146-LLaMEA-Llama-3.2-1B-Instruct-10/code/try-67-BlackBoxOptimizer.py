import random
import numpy as np
from scipy.optimize import minimize

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

    def optimize(self, func, initial_point, iterations=1000, mutation_rate=0.1):
        """
        Optimize the black box function using the BlackBoxOptimizer.

        Args:
            func (callable): The black box function to optimize.
            initial_point (List[float]): The initial point to start the optimization process.
            iterations (int, optional): The number of iterations to perform. Defaults to 1000.
            mutation_rate (float, optional): The mutation rate for the optimization process. Defaults to 0.1.

        Returns:
            List[float]: The optimized values of the function.
        """
        # Initialize the population with the initial point
        population = [initial_point]

        # Perform the specified number of iterations
        for _ in range(iterations):
            # Evaluate the fitness of each individual in the population
            fitness = [self.__call__(func, individual) for individual in population]

            # Select the fittest individuals to reproduce
            fittest_individuals = population[np.argsort(fitness)][::-1][:len(population)//2]

            # Generate a new population by mutating the fittest individuals
            new_population = [individual + random.uniform(-mutation_rate, mutation_rate) * (f - best_value) for individual, f in zip(fittest_individuals, fitness)]

            # Replace the old population with the new one
            population = new_population

        # Return the optimized values of the function
        return population