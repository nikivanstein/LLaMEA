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

    def mutate(self, individual):
        """
        Mutate the individual by changing a random value within the specified bounds.

        Args:
            individual (list): The individual to mutate.

        Returns:
            list: The mutated individual.
        """
        # Generate a random index of the bounds
        idx = np.random.randint(0, self.dim)

        # Change the value at the specified index
        individual[idx] = np.random.uniform(self.search_space[idx])

        # Return the mutated individual
        return individual

    def evolve(self, population_size, mutation_rate):
        """
        Evolve the population using the specified mutation rate.

        Args:
            population_size (int): The size of the population.
            mutation_rate (float): The mutation rate.

        Returns:
            list: The evolved population.
        """
        # Initialize the population with random individuals
        population = [self.evaluate_fitness(individual) for individual in random.sample(self.search_space, population_size)]

        # Evolve the population for the specified number of generations
        for _ in range(100):
            # Evaluate the fitness of each individual
            fitness = [self.evaluate_fitness(individual) for individual in population]

            # Select the fittest individuals
            fittest = [individual for individual, fitness in zip(population, fitness) if fitness == max(fitness)]

            # Create a new population with the fittest individuals
            new_population = [fittest[0]] + [self.mutate(individual) for individual in fittest[1:]]

            # Replace the old population with the new population
            population = new_population

            # If the mutation rate is reached, stop evolving
            if random.random() < mutation_rate:
                break

        # Return the evolved population
        return population

# One-line description with the main idea
# Novel Metaheuristic Algorithm for Black Box Optimization
# 
# This algorithm uses differential evolution to optimize the black box function, 
# refining its strategy by changing individual lines of code to refine its strategy.