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

    def mutation(self, individual):
        """
        Apply mutation to the current individual.

        Args:
            individual (list): The current individual.

        Returns:
            list: The mutated individual.
        """
        # Select a random index to mutate
        idx = np.random.randint(0, len(individual))

        # Generate a new individual by replacing the mutated index with a random value
        mutated_individual = individual[:idx] + [np.random.uniform(-5.0, 5.0)] + individual[idx+1:]

        return mutated_individual

    def crossover(self, parent1, parent2):
        """
        Perform crossover between two parents.

        Args:
            parent1 (list): The first parent.
            parent2 (list): The second parent.

        Returns:
            list: The offspring.
        """
        # Select a random index to crossover
        idx = np.random.randint(0, len(parent1))

        # Create a new offspring by combining the two parents
        offspring = parent1[:idx] + parent2[idx:]

        return offspring

    def selection(self, population):
        """
        Select the fittest individuals from the population.

        Args:
            population (list): The population.

        Returns:
            list: The fittest individuals.
        """
        # Calculate the fitness of each individual
        fitness = [self.func(individual) for individual in population]

        # Select the fittest individuals
        fittest_individuals = np.argsort(fitness)[-self.budget:]

        return fittest_individuals

    def evaluate_fitness(self, individual, budget):
        """
        Evaluate the fitness of an individual within the budget.

        Args:
            individual (list): The individual.
            budget (int): The number of function evaluations allowed.

        Returns:
            float: The fitness of the individual.
        """
        # Initialize the best value and its corresponding index
        best_value = float('-inf')
        best_index = -1

        # Perform the specified number of function evaluations
        for _ in range(budget):
            # Generate a random point in the search space
            point = self.search_space[np.random.randint(0, self.dim)]

            # Evaluate the function at the current point
            value = self.func(point)

            # If the current value is better than the best value found so far,
            # update the best value and its corresponding index
            if value > best_value:
                best_value = value
                best_index = point

        # Return the optimized value
        return best_value

# Description: Novel Metaheuristic Algorithm for Black Box Optimization (NMBA)
# Code: 