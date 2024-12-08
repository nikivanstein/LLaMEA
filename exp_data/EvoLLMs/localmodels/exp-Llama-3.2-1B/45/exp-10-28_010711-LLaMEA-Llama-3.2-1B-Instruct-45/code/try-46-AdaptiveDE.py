import numpy as np
from scipy.optimize import differential_evolution
import random

class AdaptiveDE(BBOBOptimizer):
    """
    An adaptive differential evolution algorithm for solving black box optimization problems.
    
    The algorithm adapts its strategy based on the performance of the individual lines of the selected solution.
    It uses a combination of differential evolution and adaptive mutation to search for the optimal solution in the search space.
    """

    def __init__(self, budget, dim):
        """
        Initialize the adaptive differential evolution algorithm with a budget and dimensionality.
        
        Args:
            budget (int): The number of function evaluations allowed.
            dim (int): The dimensionality of the search space.
        """
        super().__init__(budget, dim)

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
        
        # Initialize the population with random lines
        population = [random.randint(-5, 5) for _ in range(100)]
        
        # Perform the optimization using differential evolution
        for _ in range(self.budget):
            # Evaluate the fitness of each individual
            fitness = [self.evaluate_fitness(individual) for individual in population]
            
            # Select the fittest individuals
            fittest_individuals = population[np.argsort(fitness)]
            
            # Calculate the new population
            new_population = []
            for _ in range(100):
                # Select a random individual from the fittest individuals
                individual = fittest_individuals[random.randint(0, len(fittest_individuals) - 1)]
                
                # Perform differential evolution
                new_individual = self.differential_evolution(lambda x: -y, [(x, y)], x0=individual, bounds=((None, None), (None, None)), n_iter=1)
                
                # Add the new individual to the new population
                new_population.append(new_individual)
            
            # Update the population
            population = new_population
        
        # Return the optimal solution and the corresponding objective value
        return population[0], -population[0].func(x)