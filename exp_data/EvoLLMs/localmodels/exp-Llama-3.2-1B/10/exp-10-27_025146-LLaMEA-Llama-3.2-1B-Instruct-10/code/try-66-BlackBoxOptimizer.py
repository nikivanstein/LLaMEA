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

    def novel_metaheuristic(self, func, initial_population, mutation_rate, mutation_threshold, selection_method, population_size):
        """
        Novel Metaheuristic Algorithm for Black Box Optimization.

        Args:
            func (callable): The black box function to optimize.
            initial_population (list): The initial population of individuals.
            mutation_rate (float): The rate at which individuals are mutated.
            mutation_threshold (float): The threshold for mutation.
            selection_method (str): The selection method to use.
            population_size (int): The size of the population.

        Returns:
            tuple: A tuple containing the best individual and its fitness value.
        """
        # Initialize the best individual and its fitness value
        best_individual = None
        best_fitness = float('-inf')

        # Initialize the population
        population = initial_population.copy()

        # Perform the specified number of iterations
        for _ in range(1000):
            # Evaluate the fitness of each individual
            fitness = [self.__call__(func(individual)) for individual in population]

            # Select the fittest individuals using the specified selection method
            selected_individuals = selection_method(population, fitness)

            # Mutate the selected individuals
            mutated_individuals = [individual + np.random.normal(0, 1, self.dim) for individual in selected_individuals]

            # Crossover the mutated individuals
            offspring = []
            for _ in range(population_size // 2):
                parent1, parent2 = random.sample(mutated_individuals, 2)
                child = (parent1 + parent2) / 2
                offspring.append(child)

            # Mutate the offspring
            for individual in offspring:
                if random.random() < mutation_rate:
                    individual += np.random.normal(0, 1, self.dim)

            # Replace the least fit individuals with the mutated offspring
            population = [individual for individual in population if fitness.index(min(fitness)) < len(fitness) - 1] + \
                         [individual for individual in mutated_individuals if fitness.index(min(fitness)) < len(fitness) - 1]

        # Return the best individual and its fitness value
        return best_individual, best_fitness

# Description: Novel Metaheuristic Algorithm for Black Box Optimization
# Code: 