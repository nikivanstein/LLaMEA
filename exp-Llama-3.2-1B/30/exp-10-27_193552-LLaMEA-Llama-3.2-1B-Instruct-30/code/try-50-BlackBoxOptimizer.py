import random
import numpy as np

class BlackBoxOptimizer:
    """
    A metaheuristic algorithm to optimize black box functions.
    
    The algorithm uses a combination of random search, mutation, and selection to find the optimal solution.
    It handles a wide range of tasks and can be adapted to various optimization problems.
    """
    
    def __init__(self, budget, dim):
        """
        Initialize the optimizer with a budget and dimensionality.
        
        Args:
            budget (int): The maximum number of function evaluations allowed.
            dim (int): The dimensionality of the search space.
        """
        self.budget = budget
        self.dim = dim
        self.search_space = [(-5.0, 5.0)] * dim
        self.func_evaluations = 0
        self.population_size = 100
        self.mutation_rate = 0.1
        self.selection_rate = 0.3
        
    def __call__(self, func):
        """
        Optimize the black box function using the optimizer.
        
        Args:
            func (function): The black box function to optimize.
        
        Returns:
            tuple: The optimal solution and its cost.
        """
        
        # Initialize the population with random solutions
        population = self.generate_population(self.population_size, self.dim)
        
        # Run the optimizer for a specified number of generations
        for _ in range(100):
            # Evaluate the fitness of each individual
            fitness = self.evaluate_fitness(population)
            
            # Select the fittest individuals
            selected_individuals = self.select_individuals(fitness, self.population_size)
            
            # Mutate the selected individuals
            mutated_individuals = self.mutate(selected_individuals)
            
            # Replace the old population with the new one
            population = self.replace_population(population, mutated_individuals, self.budget)
            
            # Evaluate the fitness of each individual again
            fitness = self.evaluate_fitness(population)
            
            # Replace the old population with the new one
            population = self.replace_population(population, fitness, self.budget)
        
        # Return the fittest individual
        return self.select_individuals(fitness, self.population_size)[0], fitness[self.population_size - 1]
    
    def generate_population(self, size, dim):
        """
        Generate a population of random solutions.
        
        Args:
            size (int): The number of individuals in the population.
            dim (int): The dimensionality of the search space.
        
        Returns:
            list: A list of random solutions.
        """
        
        # Generate a list of random solutions
        population = [tuple(random.uniform(-5.0, 5.0), random.uniform(-5.0, 5.0)) for _ in range(size)]
        
        # Shuffle the list to introduce diversity
        random.shuffle(population)
        
        return population
    
    def select_individuals(self, fitness, size):
        """
        Select the fittest individuals.
        
        Args:
            fitness (list): A list of fitness values.
            size (int): The number of individuals to select.
        
        Returns:
            list: A list of selected individuals.
        """
        
        # Select the fittest individuals using the selection rate
        selected_individuals = [individual for _, individual in sorted(zip(fitness, population), reverse=True)[:size]]
        
        return selected_individuals
    
    def mutate(self, individuals):
        """
        Mutate the selected individuals.
        
        Args:
            individuals (list): A list of selected individuals.
        
        Returns:
            list: A list of mutated individuals.
        """
        
        # Mutate the selected individuals using the mutation rate
        mutated_individuals = [individual[:dim] + tuple(random.uniform(-5.0, 5.0), random.uniform(-5.0, 5.0)) + individual[dim:] for individual in individuals]
        
        return mutated_individuals
    
    def replace_population(self, population, fitness, budget):
        """
        Replace the old population with the new one.
        
        Args:
            population (list): The old population.
            fitness (list): The fitness values of the old population.
            budget (int): The number of function evaluations allowed.
        
        Returns:
            list: The new population.
        """
        
        # Replace the old population with the new one
        new_population = []
        for _ in range(budget):
            # Evaluate the fitness of the next individual
            fitness_value = fitness[_]
            
            # Select a random individual from the old population
            individual = population[_ % len(population)]
            
            # Evaluate the fitness of the selected individual
            fitness_value = fitness[_]
            
            # Add the selected individual to the new population
            new_population.append(individual)
            
            # Add the fitness value to the new population
            new_population[-1] += fitness_value
        
        return new_population
    
    def evaluate_fitness(self, population):
        """
        Evaluate the fitness of each individual.
        
        Args:
            population (list): A list of individuals.
        
        Returns:
            list: A list of fitness values.
        """
        
        # Evaluate the fitness of each individual
        fitness = [self.calculate_fitness(individual) for individual in population]
        
        return fitness
    
    def calculate_fitness(self, individual):
        """
        Calculate the fitness of an individual.
        
        Args:
            individual (tuple): An individual.
        
        Returns:
            float: The fitness value of the individual.
        """
        
        # Calculate the fitness value of the individual
        fitness_value = 1 / abs(individual[0] - individual[1])
        
        return fitness_value
    
# Description: Black Box Optimization Algorithm using Evolutionary Strategy
# Code: 