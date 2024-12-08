import numpy as np
from scipy.optimize import differential_evolution
import copy

class BBOBOptimizer:
    """
    A metaheuristic algorithm for solving black box optimization problems.
    
    The algorithm uses differential evolution to search for the optimal solution in the search space.
    It is designed to handle a wide range of tasks and can be tuned for different performance.
    """

    def __init__(self, budget, dim):
        """
        Initialize the optimizer with a budget and dimensionality.
        
        Args:
            budget (int): The number of function evaluations allowed.
            dim (int): The dimensionality of the search space.
        """
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        """
        Optimize a black box function using the given budget.
        
        Args:
            func (callable): The black box function to optimize.
        
        Returns:
            tuple: The optimal solution and the corresponding objective value.
        """
        # Create a grid of points in the search space
        x = np.linspace(-5.0, 5.0, self.dim)
        
        # Evaluate the black box function at each point
        y = func(x)
        
        # Initialize the population with random solutions
        population = [copy.deepcopy(x) for _ in range(100)]
        
        # Perform the optimization using differential evolution
        for _ in range(self.budget):
            # Select the fittest individual
            fittest = population[np.argmax([self.f(individual, self.logger) for individual in population])]
            
            # Create a new individual by modifying the fittest individual
            new_individual = fittest + np.random.normal(0, 1, self.dim)
            
            # Check if the new individual is within the bounds
            if np.any(new_individual < -5.0) or np.any(new_individual > 5.0):
                new_individual = fittest
            
            # Evaluate the new individual
            new_y = func(new_individual)
            
            # Update the fittest individual and the population
            population[np.argmax([self.f(individual, self.logger) for individual in population])] = new_individual
            population.append(copy.deepcopy(new_individual))
        
        # Return the optimal solution and the corresponding objective value
        return population[np.argmax([self.f(individual, self.logger) for individual in population])], -self.f(fittest, self.logger)


# Description: An adaptive differential evolution optimization algorithm.
# The algorithm uses differential evolution to search for the optimal solution in the search space.
# It is designed to handle a wide range of tasks and can be tuned for different performance.