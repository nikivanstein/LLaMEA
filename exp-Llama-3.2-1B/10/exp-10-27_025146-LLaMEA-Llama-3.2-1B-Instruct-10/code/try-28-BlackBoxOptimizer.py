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

    def novel_metaheuristic_algorithm(self, func, population_size=100, mutation_rate=0.1, bounds=None):
        """
        Novel Metaheuristic Algorithm for Black Box Optimization.

        Description: This algorithm combines the strengths of genetic algorithms and differential evolution to optimize black box functions.
        """
        # Initialize the population with random individuals
        population = np.random.uniform(self.search_space, size=(population_size, self.dim))

        # Define the fitness function to optimize
        def fitness(individual):
            # Evaluate the function at the current point
            value = func(individual)
            return -value  # Minimize the function value

        # Perform the specified number of generations
        for _ in range(100):
            # Evaluate the fitness of each individual in the population
            fitness_values = fitness(population)

            # Select the fittest individuals
            fittest_indices = np.argsort(fitness_values)[:self.budget]

            # Create a new population by mutating the fittest individuals
            new_population = np.zeros((self.budget, self.dim))
            for i in fittest_indices:
                # Randomly select a mutation point
                mutation_point = np.random.randint(0, self.dim)

                # Perform a single mutation on the current individual
                new_individual = population[i]
                new_individual[mutation_point] += random.uniform(-1, 1)
                new_individual[mutation_point] = np.clip(new_individual[mutation_point], -5.0, 5.0)

                # Add the mutated individual to the new population
                new_population[i] = new_individual

            # Replace the old population with the new population
            population = new_population

            # Update the bounds if specified
            if bounds is not None:
                for i in range(self.budget):
                    if np.any(population[i, :] < bounds[i]):
                        population[i, :] = bounds[i]
                    elif np.any(population[i, :] > bounds[i]):
                        population[i, :] = bounds[i]

        # Return the best individual in the final population
        return population[np.argmax(fitness_values)]

# Description: Novel Metaheuristic Algorithm for Black Box Optimization
# Code: 