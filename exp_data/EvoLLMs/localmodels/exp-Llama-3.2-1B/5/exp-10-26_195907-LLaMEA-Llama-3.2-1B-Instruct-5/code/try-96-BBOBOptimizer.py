import random
import numpy as np

class BBOBOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.search_space = 2 * np.random.uniform(-5.0, 5.0, size=(dim, 2))
        self.func = lambda x: np.sum(x)

    def __call__(self, func, iterations=100, mutation_rate=0.01):
        """
        Optimizes the black box function using evolutionary strategies.

        Args:
            func (callable): The black box function to optimize.
            iterations (int, optional): The number of iterations to perform. Defaults to 100.
            mutation_rate (float, optional): The rate at which to mutate the individual. Defaults to 0.01.

        Returns:
            list: A list of optimized individuals.
        """
        population = self.generate_population(iterations)
        for _ in range(iterations):
            for individual in population:
                fitness = self.evaluate_fitness(individual, func)
                if fitness < self.func(individual):
                    individual = self.mutate(individual, mutation_rate)
            population = self.select_population(population, fitness)
        return population

    def generate_population(self, iterations):
        """
        Generates a population of random individuals.

        Args:
            iterations (int): The number of iterations to perform.

        Returns:
            list: A list of individuals.
        """
        population = []
        for _ in range(iterations):
            individual = self.search_space[np.random.randint(0, self.search_space.shape[0])]
            population.append(individual)
        return population

    def select_population(self, population, fitness):
        """
        Selects a population based on fitness.

        Args:
            population (list): The population to select from.
            fitness (list): The fitness values of the individuals in the population.

        Returns:
            list: The selected population.
        """
        selected = []
        for individual in population:
            if np.random.rand() < fitness[individual]:
                selected.append(individual)
        return selected

    def mutate(self, individual, mutation_rate):
        """
        Mutates an individual with a specified rate.

        Args:
            individual (list): The individual to mutate.
            mutation_rate (float): The rate at which to mutate the individual.

        Returns:
            list: The mutated individual.
        """
        mutated = []
        for i in range(len(individual)):
            if np.random.rand() < mutation_rate:
                mutated.append(individual[i] + np.random.uniform(-1, 1))
            else:
                mutated.append(individual[i])
        return mutated

# One-line description with the main idea
# Novel Metaheuristic Algorithm for Black Box Optimization using Evolutionary Strategies
# Optimizes the black box function using evolutionary strategies, such as genetic algorithms or particle swarm optimization
# 
# ```python
# BBOBOptimizer: Novel Metaheuristic Algorithm for Black Box Optimization using Evolutionary Strategies
# ```