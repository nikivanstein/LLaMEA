import numpy as np
from scipy.optimize import minimize
import random

class AdaptiveMetaHeuristic:
    """
    An adaptive metaheuristic algorithm for solving black box optimization problems.
    
    The algorithm uses a combination of local search, genetic algorithm, and adaptive perturbation to find the optimal solution.
    """
    
    def __init__(self, budget, dim, num_generations=100):
        """
        Initializes the AdaptiveMetaHeuristic algorithm.
        
        Args:
        budget (int): The maximum number of function evaluations allowed.
        dim (int): The dimensionality of the search space.
        num_generations (int, optional): The number of generations to run the genetic algorithm. Defaults to 100.
        """
        self.budget = budget
        self.dim = dim
        self.num_generations = num_generations
        self.population_size = 100
        self.population = self.initialize_population()
        self.fitness_scores = []
        
    def initialize_population(self):
        """
        Initializes the population with random individuals.
        
        Returns:
        list: A list of individuals in the population.
        """
        return [[random.uniform(-5.0, 5.0) for _ in range(self.dim)] for _ in range(self.population_size)]
    
    def evaluate_fitness(self, individual):
        """
        Evaluates the fitness of an individual.
        
        Args:
        individual (list): The individual to evaluate.
        
        Returns:
        float: The fitness score of the individual.
        """
        func = lambda x: self.func(x)
        return func(individual)
    
    def adapt_perturbation(self, individual, bounds):
        """
        Adapts the perturbation of an individual.
        
        Args:
        individual (list): The individual to perturb.
        bounds (list): The current bounds of the search space.
        
        Returns:
        list: The perturbed individual.
        """
        # Generate a new solution by perturbing the current solution
        new_solution = [self.bounds[0] + np.random.uniform(-1, 1) * (self.bounds[1] - self.bounds[0]) for _ in range(self.dim)]
        
        # Ensure the new solution is within the bounds
        new_solution = [max(bounds[i], min(new_solution[i], bounds[i])) for i in range(self.dim)]
        
        return new_solution
    
    def genetic_algorithm(self):
        """
        Runs the genetic algorithm to find the optimal solution.
        """
        for _ in range(self.num_generations):
            # Evaluate the fitness of each individual
            self.fitness_scores = [self.evaluate_fitness(individual) for individual in self.population]
            
            # Select the fittest individuals
            self.population = self.select_fittest(self.population, self.fitness_scores)
            
            # Perform crossover and mutation
            self.population = self.crossover(self.population)
            self.population = self.mutate(self.population)
        
        # Return the fittest individual
        return self.population[0]
    
    def crossover(self, population):
        """
        Performs crossover on the population.
        
        Args:
        population (list): The population to crossover.
        
        Returns:
        list: The crosented population.
        """
        # Select the parents
        parents = random.sample(population, self.population_size // 2)
        
        # Perform crossover
        children = []
        for _ in range(self.population_size // 2):
            parent1, parent2 = random.sample(parents, 2)
            child = [self.bounds[0] + (parent1[0] + parent2[0]) / 2, self.bounds[1] + (parent1[1] + parent2[1]) / 2]
            children.append(child)
        
        return children
    
    def mutate(self, population):
        """
        Mutates the population.
        
        Args:
        population (list): The population to mutate.
        
        Returns:
        list: The mutated population.
        """
        # Select the fittest individuals
        parents = random.sample(population, self.population_size // 2)
        
        # Perform mutation
        mutated = []
        for individual in parents:
            mutated.append(self.adapt_perturbation(individual, self.bounds))
        
        return mutated
    
    def adapt_perturbation(self, individual, bounds):
        """
        Adapts the perturbation of an individual.
        
        Args:
        individual (list): The individual to perturb.
        bounds (list): The current bounds of the search space.
        
        Returns:
        list: The perturbed individual.
        """
        # Generate a new solution by perturbing the current solution
        new_solution = [self.bounds[0] + np.random.uniform(-1, 1) * (self.bounds[1] - self.bounds[0]) for _ in range(self.dim)]
        
        # Ensure the new solution is within the bounds
        new_solution = [max(bounds[i], min(new_solution[i], bounds[i])) for i in range(self.dim)]
        
        return new_solution

# Description: Black Box Optimization using MetaHeuristic Algorithm
# Code: 
# ```python
# import numpy as np
# import random
# import scipy.optimize as optimize

# def func(x):
#     return x[0]**2 + x[1]**2

# def initialize_population():
#     return np.random.uniform(-10, 10, 2)

# def evaluate_fitness(individual):
#     return func(individual)

# def adapt_perturbation(individual, bounds):
#     new_solution = [bounds[0] + np.random.uniform(-1, 1) * (bounds[1] - bounds[0]), bounds[1] + np.random.uniform(-1, 1) * (bounds[1] - bounds[0])]
#     return new_solution

# adaptive_metaheuristic = AdaptiveMetaHeuristic(100, 2)
# population = adaptive_metaheuristic.initialize_population()
# individual = np.random.uniform(-10, 10, 2)
# fitness = evaluate_fitness(individual)
# print(f"Fitness: {fitness}")
# print(f"Individual: {individual}")
# print(f"Optimal Solution: {adaptive_metaheuristic.genetic_algorithm()}")
# print(f"Optimal Fitness: {adaptive_metaheuristic.fitness_scores[-1]}")