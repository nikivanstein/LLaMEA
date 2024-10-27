import numpy as np
import random
from scipy.optimize import minimize

class AdaptiveMetaHeuristic:
    """
    An adaptive metaheuristic algorithm for solving black box optimization problems.
    
    The algorithm uses a combination of genetic algorithm and gradient-based optimization to find the optimal solution.
    
    Attributes:
    budget (int): The maximum number of function evaluations allowed.
    dim (int): The dimensionality of the search space.
    func (function): The black box function to optimize.
    search_space (list): The range of the search space.
    bounds (list): The bounds of the search space.
    population (list): The population of individuals in the genetic algorithm.
    fitness (list): The fitness of each individual in the genetic algorithm.
    """

    def __init__(self, budget, dim):
        """
        Initializes the AdaptiveMetaHeuristic algorithm.
        
        Args:
        budget (int): The maximum number of function evaluations allowed.
        dim (int): The dimensionality of the search space.
        """
        self.budget = budget
        self.dim = dim
        self.func = None
        self.search_space = None
        self.bounds = None
        self.population = None
        self.fitness = None

    def __call__(self, func):
        """
        Optimizes the black box function using AdaptiveMetaHeuristic.
        
        Args:
        func (function): The black box function to optimize.
        
        Returns:
        tuple: A tuple containing the optimal solution and its cost.
        """
        if self.func is None:
            raise ValueError("The black box function must be initialized before calling this method.")
        
        # Initialize the search space
        self.search_space = [self.bounds] * self.dim
        self.bounds = [(-5.0, 5.0)] * self.dim
        
        # Initialize the population and fitness of each individual
        self.population = self.initialize_population(func, self.budget, self.dim)
        self.fitness = self.calculate_fitness(self.population)
        
        # Perform genetic algorithm
        while self.fitness[-1] > 0:
            # Select parents using tournament selection
            parents = self.select_parents(self.population, self.budget)
            
            # Crossover (reproduce) offspring
            offspring = self.crossover(parents)
            
            # Mutate offspring
            offspring = self.mutate(offspring)
            
            # Evaluate fitness of offspring
            self.fitness = self.calculate_fitness(offspring)
            
            # Replace worst individual with offspring
            self.population = self.population[:self.budget] + offspring
    
    def initialize_population(self, func, budget, dim):
        """
        Initializes the population of individuals using tournament selection.
        
        Args:
        func (function): The black box function to optimize.
        budget (int): The maximum number of function evaluations allowed.
        dim (int): The dimensionality of the search space.
        
        Returns:
        list: The population of individuals in the genetic algorithm.
        """
        # Initialize population with random individuals
        population = [random.uniform(-5.0, 5.0) for _ in range(budget)]
        
        # Evaluate fitness of each individual
        fitness = [func(individual) for individual in population]
        
        # Select parents using tournament selection
        parents = random.choices(population, weights=fitness, k=budget//2)
        
        return parents
    
    def select_parents(self, population, budget):
        """
        Selects parents using tournament selection.
        
        Args:
        population (list): The population of individuals.
        budget (int): The maximum number of function evaluations allowed.
        
        Returns:
        list: The selected parents.
        """
        # Select parents using tournament selection
        parents = []
        for _ in range(budget):
            tournament_size = random.randint(2, budget)
            tournament = random.choices(population, weights=[fitness for individual, fitness in zip(population, fitness)], k=tournament_size)
            tournament.sort(key=lambda x: x[1], reverse=True)
            parents.append(tournament[0])
        
        return parents
    
    def crossover(self, parents):
        """
        Crossover (reproduce) offspring.
        
        Args:
        parents (list): The selected parents.
        
        Returns:
        list: The offspring.
        """
        # Perform crossover (reproduce) using uniform crossover
        offspring = []
        for _ in range(len(parents)):
            parent1, parent2 = parents[_*2], parents[_*2+1]
            offspring.append(self.reproduce(parent1, parent2))
        
        return offspring
    
    def reproduce(self, parent1, parent2):
        """
        Reproduces offspring using uniform crossover.
        
        Args:
        parent1 (list): The first parent.
        parent2 (list): The second parent.
        
        Returns:
        list: The offspring.
        """
        # Select crossover point
        crossover_point = random.randint(0, len(parent1)-1)
        
        # Perform crossover (reproduce) using uniform crossover
        offspring = parent1[:crossover_point] + parent2[crossover_point:]
        
        return offspring
    
    def mutate(self, offspring):
        """
        Mutates offspring using mutation.
        
        Args:
        offspring (list): The offspring.
        
        Returns:
        list: The mutated offspring.
        """
        # Perform mutation using uniform mutation
        mutated_offspring = offspring[:]
        for _ in range(len(offspring)):
            mutation_point = random.randint(0, len(offspring)-1)
            mutated_offspring[mutation_point] += random.uniform(-1, 1)
        
        return mutated_offspring
    
    def calculate_fitness(self, population):
        """
        Evaluates fitness of each individual.
        
        Args:
        population (list): The population of individuals.
        
        Returns:
        list: The fitness of each individual.
        """
        fitness = []
        for individual in population:
            fitness.append(func(individual))
        
        return fitness
    
    def evaluate_fitness(self, individual, func):
        """
        Evaluates fitness of an individual using the given function.
        
        Args:
        individual (float): The individual to evaluate.
        func (function): The function to evaluate.
        
        Returns:
        float: The fitness of the individual.
        """
        return func(individual)