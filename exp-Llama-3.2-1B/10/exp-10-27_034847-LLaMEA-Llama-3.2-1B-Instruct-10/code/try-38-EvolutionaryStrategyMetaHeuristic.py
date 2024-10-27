import numpy as np
from scipy.optimize import minimize

class EvolutionaryStrategyMetaHeuristic:
    """
    An evolutionary strategy optimization algorithm for solving black box optimization problems.
    
    The algorithm uses a combination of evolutionary strategies and gradient-based optimization to find the optimal solution.
    
    Attributes:
    budget (int): The maximum number of function evaluations allowed.
    dim (int): The dimensionality of the search space.
    bounds (list): The bounds of the search space.
    population_size (int): The size of the population.
    mutation_rate (float): The rate at which the population is mutated.
    """

    def __init__(self, budget, dim, population_size, mutation_rate):
        """
        Initializes the EvolutionaryStrategyMetaHeuristic algorithm.
        
        Args:
        budget (int): The maximum number of function evaluations allowed.
        dim (int): The dimensionality of the search space.
        population_size (int): The size of the population.
        mutation_rate (float): The rate at which the population is mutated.
        """
        self.budget = budget
        self.dim = dim
        self.bounds = None
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.population = None

    def __call__(self, func):
        """
        Optimizes the black box function using EvolutionaryStrategyMetaHeuristic.
        
        Args:
        func (function): The black box function to optimize.
        
        Returns:
        tuple: A tuple containing the optimal solution and its cost.
        """
        if self.bounds is None:
            raise ValueError("The search space must be initialized before calling this method.")
        
        # Initialize the search space
        self.bounds = [(-5.0, 5.0)] * self.dim
        
        # Initialize the population
        self.population = np.random.uniform(self.bounds[0], self.bounds[1], size=(self.population_size, self.dim))
        
        # Initialize the optimal solution and its cost
        opt_solution = None
        opt_cost = float('inf')
        
        # Perform evolutionary strategy optimization
        for _ in range(self.budget):
            # Evaluate the population using the black box function
            fitness = self.evaluate_fitness(self.population)
            
            # Select the fittest individual
            self.population = self.select_fittest_individual(fitness)
            
            # Mutate the population
            self.population = self.mutate_population(self.population, self.mutation_rate)
            
            # Update the optimal solution and its cost if necessary
            if self.evaluate_fitness(self.population) < opt_cost:
                opt_solution = self.population
                opt_cost = self.evaluate_fitness(self.population)
        
        # Return the optimal solution and its cost
        return opt_solution, opt_cost

    def evaluate_fitness(self, population):
        """
        Evaluates the fitness of the population using the black box function.
        
        Args:
        population (numpy.ndarray): The population to evaluate.
        
        Returns:
        float: The fitness of the population.
        """
        # Evaluate the fitness of each individual in the population
        fitness = np.array([self.func(individual) for individual in population])
        
        # Return the average fitness
        return np.mean(fitness)

    def select_fittest_individual(self, fitness):
        """
        Selects the fittest individual in the population using tournament selection.
        
        Args:
        fitness (numpy.ndarray): The fitness of the population.
        
        Returns:
        numpy.ndarray: The fittest individual in the population.
        """
        # Select the fittest individual using tournament selection
        tournament_size = self.population_size // 2
        winners = np.random.choice(range(self.population_size), size=tournament_size, replace=False, p=fitness / np.sum(fitness))
        return np.array([self.population[i] for i in winners])

    def mutate_population(self, population, mutation_rate):
        """
        Mutates the population using bit-flipping.
        
        Args:
        population (numpy.ndarray): The population to mutate.
        mutation_rate (float): The rate at which the population is mutated.
        
        Returns:
        numpy.ndarray: The mutated population.
        """
        # Mutate the population using bit-flipping
        mutated_population = np.copy(population)
        for i in range(mutated_population.shape[0]):
            mutated_population[i] = mutated_population[i] ^ (np.random.uniform(0, 1) < mutation_rate)
        
        return mutated_population