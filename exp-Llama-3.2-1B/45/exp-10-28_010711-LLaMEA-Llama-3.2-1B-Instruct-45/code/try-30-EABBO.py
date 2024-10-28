import numpy as np
from scipy.optimize import differential_evolution
import random

class EABBO:
    """
    An evolutionary algorithm for black box optimization problems.
    
    The algorithm uses differential evolution to search for the optimal solution in the search space.
    It is designed to handle a wide range of tasks and can be tuned for different performance.
    """

    def __init__(self, budget, dim):
        """
        Initialize the evolutionary algorithm with a budget and dimensionality.
        
        Args:
            budget (int): The number of function evaluations allowed.
            dim (int): The dimensionality of the search space.
        """
        self.budget = budget
        self.dim = dim

    def __call__(self, func, population_size=100, mutation_rate=0.01):
        """
        Optimize a black box function using the given budget and population size.
        
        Args:
            func (callable): The black box function to optimize.
            population_size (int): The size of the population.
            mutation_rate (float): The rate of mutation.
        
        Returns:
            tuple: The optimal solution and the corresponding objective value.
        """
        # Create a population of random individuals
        population = self.generate_population(population_size, dim)
        
        # Perform the optimization using differential evolution
        for _ in range(self.budget):
            # Evaluate the fitness of each individual
            fitness = [self.evaluate_fitness(individual, func) for individual in population]
            
            # Select the fittest individuals
            fittest_individuals = population[np.argsort(fitness)[:self.population_size//2]]
            
            # Create a new population by mutating the fittest individuals
            new_population = self.mutate(population, fittest_individuals, mutation_rate)
            
            # Replace the old population with the new one
            population = new_population
        
        # Return the optimal solution and the corresponding objective value
        return self.evaluate_fitness(population[0], func), -np.max(fitness)

    def generate_population(self, population_size, dim):
        """
        Generate a population of random individuals.
        
        Args:
            population_size (int): The size of the population.
            dim (int): The dimensionality of the search space.
        
        Returns:
            list: The population of random individuals.
        """
        return [random.uniform(-5.0, 5.0) for _ in range(population_size)]

    def mutate(self, population, fittest_individuals, mutation_rate):
        """
        Mutate the fittest individuals in the population.
        
        Args:
            population (list): The population of individuals.
            fittest_individuals (list): The fittest individuals.
            mutation_rate (float): The rate of mutation.
        
        Returns:
            list: The mutated population.
        """
        mutated_population = []
        for individual in population:
            # Select a random mutation point
            mutation_point = random.randint(0, dim-1)
            
            # Create a new individual by modifying the current one
            new_individual = individual.copy()
            new_individual[mutation_point] += random.uniform(-1.0, 1.0)
            
            # If the mutation point is in the fittest individuals, mutate them
            if mutation_point in fittest_individuals:
                new_individuals = [new_individual]
            else:
                new_individuals = [new_individual]
            
            # Replace the old individuals with the new ones
            mutated_population.extend(new_individuals)
        
        # Replace the old population with the new one
        population = mutated_population

    def evaluate_fitness(self, individual, func):
        """
        Evaluate the fitness of an individual.
        
        Args:
            individual (tuple): The individual to evaluate.
            func (callable): The black box function to evaluate.
        
        Returns:
            float: The fitness of the individual.
        """
        return func(individual)


# Description: Evolutionary Algorithm for Black Box Optimization (EA-BBO)
# Code: 