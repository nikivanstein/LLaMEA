import numpy as np
import random
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

    def mutate(self, individual):
        """
        Mutate an individual by changing a single gene.

        Args:
            individual (list): The individual to mutate.

        Returns:
            list: The mutated individual.
        """
        # Randomly select a gene to mutate
        index = np.random.randint(0, self.dim)

        # Generate a new gene by flipping the bit at the selected index
        new_gene = individual[:index] + [1 - individual[index]] + individual[index + 1:]

        # Return the mutated individual
        return new_gene

    def anneal(self, initial_value, final_value, temperature):
        """
        Simulated annealing to find the optimal solution.

        Args:
            initial_value (float): The initial temperature.
            final_value (float): The final temperature.
            temperature (float): The current temperature.

        Returns:
            float: The optimal value.
        """
        # Initialize the temperature
        temperature = initial_value

        # Perform the specified number of iterations
        for _ in range(self.budget):
            # Generate a new point using the current temperature
            point = np.random.uniform(-5.0, 5.0, self.dim)

            # Evaluate the function at the current point
            value = self.func(point)

            # If the current value is better than the optimal value found so far,
            # accept the new point with probability exp((current_value - optimal_value) / temperature)
            if value > self.func(point):
                point = point

            # Accept the new point with probability exp((current_value - optimal_value) / temperature)
            if np.random.rand() < np.exp((value - self.func(point)) / temperature):
                point = self.mutate(point)

            # Return the optimal value
            return point

    def optimize(self, func):
        """
        Optimize the black box function using a combination of genetic algorithm and simulated annealing.

        Args:
            func (callable): The black box function to optimize.

        Returns:
            float: The optimized value of the function.
        """
        # Initialize the population
        population = [self.evaluate_fitness(func) for _ in range(100)]

        # Evolve the population for the specified number of generations
        for _ in range(100):
            # Generate the next generation
            next_generation = []
            for individual in population:
                # Generate a new individual using the annealing algorithm
                individual = self.anneal(0, 100, temperature=0.1)

                # Evaluate the function at the current individual
                value = self.func(individual)

                # Add the individual to the next generation
                next_generation.append(value)

            # Replace the old population with the new generation
            population = next_generation

        # Return the best individual in the final generation
        return population[0]