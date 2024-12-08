# Description: Novel Metaheuristic Algorithm for Black Box Optimization
# Code: 
# ```python
import random
import numpy as np
from scipy.optimize import differential_evolution

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

    def optimize(self, func, initial_population, budget):
        """
        Optimize the black box function using a population-based approach.

        Args:
            func (callable): The black box function to optimize.
            initial_population (list): The initial population of individuals.
            budget (int): The maximum number of function evaluations allowed.

        Returns:
            list: The optimized population.
        """
        # Initialize the population with random points in the search space
        population = [initial_population]

        # Perform the specified number of function evaluations
        for _ in range(budget):
            # Evaluate the function at each point in the population
            for individual in population:
                value = func(individual)

                # Select the fittest individual
                fittest_index = np.argmax([individual, value])
                fittest_individual = population[fittest_index]

                # Mutate the fittest individual
                mutated_individual = fittest_individual + np.random.normal(0, 1, self.dim)

                # Evaluate the function at the mutated individual
                mutated_value = func(mutated_individual)

                # If the mutated value is better than the best value found so far,
                # update the best value and its corresponding index
                if mutated_value > best_value:
                    best_value = mutated_value
                    best_index = fittest_index

            # Add the best individual to the population
            population.append(fittest_individual)

        # Return the optimized population
        return population

# Example usage:
budget = 1000
dim = 5
func = lambda x: x[0]**2 + x[1]**2
initial_population = [[-1.0, -1.0], [-2.0, -2.0], [-3.0, -3.0], [-4.0, -4.0], [-5.0, -5.0]]
optimized_population = BlackBoxOptimizer(budget, dim).optimize(func, initial_population, budget)

# Print the optimized population
print(optimized_population)