# Description: Novel Metaheuristic Algorithm for Black Box Optimization
# Code: 
# ```python
import random
import numpy as np
from scipy.optimize import minimize
from copy import deepcopy

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

    def mutate(self, individual):
        """
        Randomly mutate an individual in the search space.

        Args:
            individual (List[float]): The individual to mutate.

        Returns:
            List[float]: The mutated individual.
        """
        # Select a random index to mutate
        index = np.random.randint(0, self.dim)

        # Randomly change the value at the selected index
        mutated_individual = individual.copy()
        mutated_individual[index] = random.uniform(self.search_space[index])

        return mutated_individual

    def crossover(self, parent1, parent2):
        """
        Perform crossover between two parents to create a new child.

        Args:
            parent1 (List[float]): The first parent.
            parent2 (List[float]): The second parent.

        Returns:
            List[float]: The child.
        """
        # Select a random index to crossover
        index = np.random.randint(0, self.dim)

        # Create a new child by combining the values at the selected index
        child = parent1.copy()
        child[index] = parent2[index]

        return child

    def evolve(self, population, mutation_rate, crossover_rate):
        """
        Evolve a population of individuals using mutation and crossover.

        Args:
            population (List[List[float]]): The population of individuals.
            mutation_rate (float): The probability of mutation.
            crossover_rate (float): The probability of crossover.

        Returns:
            List[List[float]]: The evolved population.
        """
        # Initialize the new population
        new_population = []

        # Iterate over the population
        for i in range(len(population)):
            # Generate a new individual by mutating and crossover
            new_individual = self.mutate(population[i])

            # Crossover the new individual with another parent
            if np.random.rand() < crossover_rate:
                parent1, parent2 = population[i], population[np.random.randint(0, len(population))]
                new_individual = self.crossover(parent1, parent2)

            # Add the new individual to the new population
            new_population.append(new_individual)

        return new_population

# One-line description: Novel Metaheuristic Algorithm for Black Box Optimization
# Code: 
# ```python
# Novel Metaheuristic Algorithm for Black Box Optimization
# ```
# ```python
def optimize_bbo(budget, dim):
    """
    Optimize a black box function using the Novel Metaheuristic Algorithm.

    Args:
        budget (int): The maximum number of function evaluations allowed.
        dim (int): The dimensionality of the search space.

    Returns:
        float: The optimized value of the function.
    """
    # Initialize the BlackBoxOptimizer
    optimizer = BlackBoxOptimizer(budget, dim)

    # Evolve the population using mutation and crossover
    population = [optimizer.__call__(func) for func in globals()]
    new_population = optimizer.evolve(population, 0.1, 0.5)

    # Return the best value found
    return max(new_population)

# Example usage:
func = lambda x: x**2
budget = 100
dim = 10

# Optimize the function
optimized_value = optimize_bbo(budget, dim)
print(f"Optimized value: {optimized_value}")