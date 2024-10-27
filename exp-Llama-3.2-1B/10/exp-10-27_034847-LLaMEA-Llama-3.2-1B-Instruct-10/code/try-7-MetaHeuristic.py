import numpy as np
import random
from scipy.optimize import minimize

class MetaHeuristic:
    """
    A metaheuristic algorithm for solving black box optimization problems.
    
    The algorithm uses a combination of local search, genetic algorithm, and gradient-based optimization to find the optimal solution.
    
    Attributes:
    budget (int): The maximum number of function evaluations allowed.
    dim (int): The dimensionality of the search space.
    func (function): The black box function to optimize.
    search_space (list): The range of the search space.
    bounds (list): The bounds of the search space.
    population (list): The population of individuals.
    fitness (list): The fitness of each individual.
    """

    def __init__(self, budget, dim):
        """
        Initializes the MetaHeuristic algorithm.
        
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
        Optimizes the black box function using MetaHeuristic.
        
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
        
        # Initialize the population
        self.population = [[random.uniform(-5.0, 5.0) for _ in range(self.dim)] for _ in range(100)]
        
        # Initialize the fitness
        self.fitness = [self.func(individual) for individual in self.population]
        
        # Perform genetic algorithm
        for _ in range(self.budget):
            # Select parents
            parents = self.select_parents(self.population)
            
            # Crossover (reproduce)
            children = self.crossover(parents)
            
            # Mutate (perturb)
            children = self.mutate(children)
            
            # Evaluate fitness
            self.population = [self.evaluate_fitness(individual) for individual in children]
        
        # Return the optimal solution and its cost
        return self.population[0], self.fitness[0]

    def select_parents(self, population):
        """
        Selects parents using tournament selection.
        
        Args:
        population (list): The population of individuals.
        
        Returns:
        list: The selected parents.
        """
        # Select parents using tournament selection
        parents = []
        for _ in range(10):
            parent = random.choice(population)
            winner = max(population, key=lambda individual: self.evaluate_fitness(individual))
            parents.append(winner)
        
        return parents

    def crossover(self, parents):
        """
        Performs crossover (reproduce) using uniform crossover.
        
        Args:
        parents (list): The parents.
        
        Returns:
        list: The offspring.
        """
        # Perform crossover (reproduce) using uniform crossover
        offspring = []
        for _ in range(len(parents)):
            parent1, parent2 = random.sample(parents, 2)
            child = [self.bounds[0] + (self.bounds[1] - self.bounds[0]) * random.uniform(-1, 1) for _ in range(self.dim)]
            offspring.append(child)
        
        return offspring

    def mutate(self, children):
        """
        Mutates the offspring using bit-flipping.
        
        Args:
        children (list): The offspring.
        
        Returns:
        list: The mutated offspring.
        """
        # Mutate the offspring using bit-flipping
        mutated_children = []
        for child in children:
            mutated_child = [self.bounds[0] + (self.bounds[1] - self.bounds[0]) * random.uniform(-1, 1) for _ in range(self.dim)]
            mutated_children.append(mutated_child)
        
        return mutated_children

    def evaluate_fitness(self, individual):
        """
        Evaluates the fitness of an individual.
        
        Args:
        individual (list): The individual.
        
        Returns:
        float: The fitness of the individual.
        """
        # Evaluate the fitness of the individual
        fitness = self.func(individual)
        return fitness

# Description: Evolutionary Optimization using Genetic Algorithm
# Code: 
# ```python
# MetaHeuristic: Evolutionary Optimization using Genetic Algorithm
# Code: 
# ```python
# ```python
# import numpy as np
# import random
# import matplotlib.pyplot as plt
# from scipy.optimize import minimize

# class MetaHeuristic:
#     """
#     A metaheuristic algorithm for solving black box optimization problems.
#     """
#     def __init__(self, budget, dim):
#         """
#         Initializes the MetaHeuristic algorithm.
#         
#         Args:
#         budget (int): The maximum number of function evaluations allowed.
#         dim (int): The dimensionality of the search space.
#         """
#         self.budget = budget
#         self.dim = dim
#         self.func = None
#         self.search_space = None
#         self.bounds = None
#         self.population = None
#         self.fitness = None

#     def __call__(self, func):
#         """
#         Optimizes the black box function using MetaHeuristic.
#         
#         Args:
#         func (function): The black box function to optimize.
#         
#         Returns:
#         tuple: A tuple containing the optimal solution and its cost.
#         """
#         if self.func is None:
#             raise ValueError("The black box function must be initialized before calling this method.")
# 
#         # Initialize the search space
#         self.search_space = [self.bounds] * self.dim
#         self.bounds = [(-5.0, 5.0)] * self.dim
        
#         # Initialize the population
#         self.population = [[random.uniform(-5.0, 5.0) for _ in range(self.dim)] for _ in range(100)]
# 
#         # Initialize the fitness
#         self.fitness = [self.func(individual) for individual in self.population]
# 
#         # Perform genetic algorithm
#         for _ in range(self.budget):
#             # Select parents
#             parents = self.select_parents(self.population)
# 
#             # Crossover (reproduce)
#             children = self.crossover(parents)
# 
#             # Mutate (perturb)
#             children = self.mutate(children)
# 
#             # Evaluate fitness
#             self.population = [self.evaluate_fitness(individual) for individual in children]
# 
#         # Return the optimal solution and its cost
#         return self.population[0], self.fitness[0]

# class GeneticAlgorithm:
#     """
#     A genetic algorithm for solving black box optimization problems.
#     """
#     def __init__(self, budget, dim):
#         """
#         Initializes the GeneticAlgorithm algorithm.
#         
#         Args:
#         budget (int): The maximum number of function evaluations allowed.
#         dim (int): The dimensionality of the search space.
#         """
#         self.budget = budget
#         self.dim = dim
# 
#     def __call__(self, func):
#         """
#         Optimizes the black box function using GeneticAlgorithm.
#         
#         Args:
#         func (function): The black box function to optimize.
#         
#         Returns:
#         tuple: A tuple containing the optimal solution and its cost.
#         """
#         # Initialize the population
#         self.population = [[random.uniform(-5.0, 5.0) for _ in range(self.dim)] for _ in range(100)]
# 
#         # Initialize the fitness
#         self.fitness = [self.func(individual) for individual in self.population]
# 
#         # Perform selection
#         self.population = self.select_population()
# 
#         # Perform crossover and mutation
#         self.population = self.crossover_and_mutate(self.population)
# 
#         # Return the optimal solution and its cost
#         return self.population[0], self.fitness[0]

# def select_parents(population):
#     """
#     Selects parents using tournament selection.
#     
#     Args:
#     population (list): The population of individuals.
#     
#     Returns:
#     list: The selected parents.
#     """
#     # Select parents using tournament selection
#     parents = []
#     for _ in range(10):
#         parent = random.choice(population)
#         winner = max(population, key=lambda individual: self.evaluate_fitness(individual))
#         parents.append(winner)
# 
#     return parents

# def crossover(parents):
#     """
#     Performs crossover (reproduce) using uniform crossover.
#     
#     Args:
#     parents (list): The parents.
#     
#     Returns:
#     list: The offspring.
#     """
#     # Perform crossover (reproduce) using uniform crossover
#     offspring = []
#     for _ in range(len(parents)):
#         parent1, parent2 = random.sample(parents, 2)
#         child = [self.bounds[0] + (self.bounds[1] - self.bounds[0]) * random.uniform(-1, 1) for _ in range(self.dim)]
#         offspring.append(child)
# 
#     return offspring

# def mutate(children):
#     """
#     Mutates the offspring using bit-flipping.
#     
#     Args:
#     children (list): The offspring.
#     
#     Returns:
#     list: The mutated offspring.
#     """
#     # Mutate the offspring using bit-flipping
#     mutated_children = []
#     for child in children:
#         mutated_child = [self.bounds[0] + (self.bounds[1] - self.bounds[0]) * random.uniform(-1, 1) for _ in range(self.dim)]
#         mutated_children.append(mutated_child)
# 
#     return mutated_children

# def evaluate_fitness(individual):
#     """
#     Evaluates the fitness of an individual.
#     
#     Args:
#     individual (list): The individual.
#     
#     Returns:
#     float: The fitness of the individual.
#     """
#     # Evaluate the fitness of the individual
#     fitness = self.func(individual)
#     return fitness

# def select_population(self):
#     """
#     Selects the population for the next generation.
#     
#     Returns:
#     list: The selected population.
#     """
#     # Select the population for the next generation
#     return self.population

# def crossover_and_mutate(self, population):
#     """
#     Performs crossover and mutation on the population.
#     
#     Returns:
#     list: The mutated population.
#     """
#     # Perform crossover and mutation on the population
#     mutated_population = []
#     for individual in population:
#         mutated_individual = [self.bounds[0] + (self.bounds[1] - self.bounds[0]) * random.uniform(-1, 1) for _ in range(self.dim)]
#         mutated_individual = self.mutate(mutated_individual)
#         mutated_population.append(mutated_individual)
# 
#     return mutated_population

# def genetic_algorithm(func, budget, dim):
#     """
#     Runs a genetic algorithm to optimize the black box function.
#     
#     Args:
#     func (function): The black box function to optimize.
#     budget (int): The maximum number of function evaluations allowed.
#     dim (int): The dimensionality of the search space.
#     """
#     # Initialize the MetaHeuristic algorithm
#     metaheuristic = MetaHeuristic(budget, dim)
# 
#     # Run the genetic algorithm
#     metaheuristic.population = [random.uniform(-5.0, 5.0) for _ in range(100)]
#     metaheuristic.fitness = [func(individual) for individual in metaheuristic.population]
# 
#     # Run the genetic algorithm
#     while metaheuristic.fitness[0] > -100:
#         # Select parents
#         parents = select_parents(metaheuristic.population)
# 
#         # Crossover and mutation
#         offspring = crossover(parents)
#         metaheuristic.population = self.mutate(offspring)
# 
#         # Evaluate fitness
#         metaheuristic.fitness = [func(individual) for individual in metaheuristic.population]
# 
#         # Print the results
#         print(f"Optimal solution: {metaheuristic.population[0]} with fitness: {metaheuristic.fitness[0]}")

# def main():
#     # Define the black box function
#     def func(individual):
#         return individual[0]**2 + individual[1]**2
    
#     # Run the genetic algorithm
#     genetic_algorithm(func, 1000, 5)

# main()