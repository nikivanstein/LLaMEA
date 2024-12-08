# Description: Novel Metaheuristic Algorithm for Black Box Optimization (NMABA)
# Code: 
# ```python
import random
import numpy as np
from collections import deque
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
        self.population = deque(maxlen=100)
        self.population_dict = {}

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
        Mutate an individual in the population.

        Args:
            individual (list[float]): The individual to mutate.

        Returns:
            list[float]: The mutated individual.
        """
        # Randomly select two points in the search space
        p1, p2 = random.sample(range(self.dim), 2)

        # Swap the two points
        individual[p1], individual[p2] = individual[p2], individual[p1]

        # Return the mutated individual
        return individual

    def crossover(self, parent1, parent2):
        """
        Perform crossover between two parents to generate a new individual.

        Args:
            parent1 (list[float]): The first parent.
            parent2 (list[float]): The second parent.

        Returns:
            list[float]: The new individual.
        """
        # Randomly select a point in the search space
        p = np.random.randint(0, self.dim)

        # Create a new individual by replacing the point with the values from the parents
        child = parent1[:p] + parent2[p:]

        # Return the new individual
        return child

    def evaluate_fitness(self, individual, budget):
        """
        Evaluate the fitness of an individual.

        Args:
            individual (list[float]): The individual to evaluate.
            budget (int): The number of function evaluations allowed.

        Returns:
            float: The fitness of the individual.
        """
        # Initialize the best value and its corresponding index
        best_value = float('-inf')
        best_index = -1

        # Perform the specified number of function evaluations
        for _ in range(min(budget, self.budget)):
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

    def select(self, fitness):
        """
        Select the individual with the best fitness.

        Args:
            fitness (float): The fitness of the individual.

        Returns:
            list[float]: The selected individual.
        """
        # Select the individual with the best fitness
        selected_individual = [individual for individual, fitness in zip(self.population, fitness) if fitness == max(fitness)]

        # Return the selected individual
        return selected_individual[0]

    def update(self, fitness):
        """
        Update the population with a new individual.

        Args:
            fitness (float): The fitness of the individual.
        """
        # Select the individual with the best fitness
        selected_individual = self.select(fitness)

        # Create a new individual by replacing the selected individual with a random individual
        new_individual = self.crossover(selected_individual, self.population[-1])

        # Update the population
        self.population.append(new_individual)

        # Remove the selected individual from the population
        self.population.pop()

        # Update the population dictionary
        self.population_dict[selected_individual] = fitness

    def func(self, point):
        """
        Evaluate the function at a point.

        Args:
            point (list[float]): The point to evaluate the function at.

        Returns:
            float: The value of the function at the point.
        """
        # Evaluate the function at the point
        value = np.sum(point**2)

        # Return the value of the function
        return value

# Description: Novel Metaheuristic Algorithm for Black Box Optimization (NMABA)
# Code: 
# ```python
black_box_optimizer = BlackBoxOptimizer(1000, 10)
# black_box_optimizer.update(100)
# print(black_box_optimizer.population_dict)