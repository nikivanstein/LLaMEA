import random
import numpy as np

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

        # Initialize the population of individuals
        population = self.initialize_population(func, self.budget, self.dim)

        # Perform the specified number of function evaluations
        for _ in range(self.budget):
            # Evaluate the fitness of each individual in the population
            fitness = [self.evaluate_fitness(individual, func) for individual in population]

            # Select the fittest individuals
            fittest_individuals = np.argsort(fitness)[::-1][:self.dim]

            # Create a new population by refining the selected individuals
            new_population = self.refine_population(population, fittest_individuals, self.search_space)

            # Replace the old population with the new one
            population = new_population

            # Update the best value and its corresponding index
            best_value = max(best_value, max(fitness))
            best_index = np.argmax(fitness)

        # Return the optimized value
        return best_value

    def initialize_population(self, func, budget, dim):
        """
        Initialize the population of individuals.

        Args:
            func (callable): The black box function to optimize.
            budget (int): The maximum number of function evaluations allowed.
            dim (int): The dimensionality of the search space.

        Returns:
            list: The population of individuals.
        """
        population = []
        for _ in range(budget):
            # Generate a random point in the search space
            point = self.search_space[np.random.randint(0, self.dim)]

            # Evaluate the function at the current point
            value = func(point)

            # Add the individual to the population
            population.append(point)

        return population

    def refine_population(self, population, fittest_individuals, search_space):
        """
        Refine the population of individuals.

        Args:
            population (list): The population of individuals.
            fittest_individuals (list): The fittest individuals.
            search_space (numpy array): The search space.

        Returns:
            list: The refined population of individuals.
        """
        refined_population = population.copy()
        for individual in fittest_individuals:
            # Evaluate the fitness of the individual
            fitness = self.evaluate_fitness(individual, func)

            # Refine the individual by adjusting its step size
            step_size = self.adjust_step_size(individual, fitness, search_space)

            # Update the individual in the refined population
            refined_population.remove(individual)
            refined_population.append(individual + [step_size])

        return refined_population

    def adjust_step_size(self, individual, fitness, search_space):
        """
        Adjust the step size of an individual.

        Args:
            individual (list): The individual to adjust.
            fitness (float): The fitness of the individual.
            search_space (numpy array): The search space.

        Returns:
            list: The adjusted step size.
        """
        step_size = 0
        for i in range(len(individual) - 1):
            # Calculate the difference between the current and next points
            difference = search_space[i + 1] - search_space[i]

            # Calculate the step size
            step_size += difference * (fitness - individual[i])

            # Limit the step size to a reasonable value
            step_size = max(-10, min(step_size, 10))

        return step_size