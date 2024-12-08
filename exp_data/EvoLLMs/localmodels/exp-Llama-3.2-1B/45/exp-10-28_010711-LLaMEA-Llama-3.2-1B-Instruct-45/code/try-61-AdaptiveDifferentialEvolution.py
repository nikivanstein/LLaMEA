import numpy as np
from scipy.optimize import differential_evolution
import random

class AdaptiveDifferentialEvolution(BBOBOptimizer):
    """
    An adaptive differential evolution algorithm for solving black box optimization problems.
    
    The algorithm uses differential evolution to search for the optimal solution in the search space.
    It adapts its strategy based on the performance of the individual lines of the selected solution.
    """

    def __init__(self, budget, dim):
        """
        Initialize the adaptive differential evolution optimizer with a budget and dimensionality.
        
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

        # Initialize the population with random individual lines
        population = self.evaluate_fitness(np.random.uniform(-5.0, 5.0, self.dim), self.logger)

        # Initialize the best solution and its fitness
        best_individual = None
        best_fitness = float('inf')

        # Run the differential evolution algorithm
        for _ in range(self.budget):
            # Select the individual lines with the highest fitness
            selected_individuals = sorted(population, key=lambda x: x[-1], reverse=True)[:self.dim]

            # Initialize the new population with the selected individual lines
            new_population = selected_individuals

            # Evaluate the objective function for the new population
            new_fitness = np.array([self.evaluate_fitness(individual, self.logger) for individual in new_population])

            # Calculate the average fitness of the new population
            new_fitness_avg = np.mean(new_fitness)

            # If the average fitness is better than the best fitness, update the best solution
            if new_fitness_avg < best_fitness:
                best_individual = new_population[0]
                best_fitness = new_fitness_avg

            # Refine the strategy by changing the proportion of individual lines
            proportion = (best_fitness / new_fitness_avg) ** 0.5
            if random.random() < proportion:
                new_population = selected_individuals

            # Update the population with the new individual lines
            population = new_population

        # Return the optimal solution and the corresponding objective value
        return best_individual, -best_fitness