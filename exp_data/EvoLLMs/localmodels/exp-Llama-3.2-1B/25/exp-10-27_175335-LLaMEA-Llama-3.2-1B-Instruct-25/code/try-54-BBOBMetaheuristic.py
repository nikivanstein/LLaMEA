import numpy as np
from collections import deque
from copy import deepcopy

class BBOBMetaheuristic:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.func_evals = 0
        self.population = []

    def __call__(self, func):
        # Check if the function can be evaluated within the budget
        if self.func_evals >= self.budget:
            raise ValueError("Not enough evaluations left to optimize the function")

        # Evaluate the function within the budget
        func_evals = self.func_evals
        self.func_evals += 1
        return func

    def search(self, func):
        # Define the search space
        bounds = np.linspace(-5.0, 5.0, self.dim, endpoint=False)
        
        # Initialize the solution
        sol = None
        
        # Try different initializations
        for _ in range(10):
            # Randomly initialize the solution
            sol = np.random.uniform(bounds, size=self.dim)
            
            # Evaluate the function at the solution
            func_sol = self.__call__(func, sol)
            
            # Check if the solution is better than the current best
            if func_sol < self.__call__(func, sol):
                # Update the solution
                sol = sol
        
        # Return the best solution found
        return sol

    def mutate(self, individual):
        # Randomly select a mutation point
        idx = np.random.randint(0, self.dim)
        
        # Swap the elements at the mutation point with two other elements
        new_individual = individual.copy()
        new_individual[idx], new_individual[np.random.randint(0, self.dim)] = new_individual[np.random.randint(0, self.dim)], new_individual[idx]
        
        # Update the mutation counter
        self.population.append(new_individual)

    def evolve(self, population):
        # Initialize a new population
        new_population = deque()
        
        # Perform crossover
        for _ in range(len(population) // 2):
            parent1 = np.random.choice(population, size=self.dim, replace=False)
            parent2 = np.random.choice(population, size=self.dim, replace=False)
            child = (parent1 + parent2) / 2
            
            # Ensure the child is within the bounds
            child = np.clip(child, -5.0, 5.0)
            
            # Add the child to the new population
            new_population.append(child)
        
        # Perform mutation
        for individual in new_population:
            if np.random.rand() < 0.25:
                self.mutate(individual)
        
        # Replace the old population with the new population
        self.population = list(new_population)

    def evaluate_fitness(self, func, population):
        # Initialize a dictionary to store the fitness scores
        fitness_scores = {}
        
        # Evaluate the function for each individual in the population
        for individual in population:
            func_sol = self.__call__(func, individual)
            fitness_scores[individual] = func_sol
        
        # Return the fitness scores
        return fitness_scores

# Description: Evolutionary Algorithm for Black Box Optimization using Genetic Programming
# Code: 
# ```python
# import numpy as np
# import random

# class BBOBMetaheuristic:
#     def __init__(self, budget, dim):
#         self.budget = budget
#         self.dim = dim
#         self.func_evals = 0
#         self.population = []

#     def __call__(self, func):
#         # Check if the function can be evaluated within the budget
#         if self.func_evals >= self.budget:
#             raise ValueError("Not enough evaluations left to optimize the function")

#         # Evaluate the function within the budget
#         func_evals = self.func_evals
#         self.func_evals += 1
#         return func

#     def search(self, func):
#         # Define the search space
#         bounds = np.linspace(-5.0, 5.0, self.dim, endpoint=False)
        
#         # Initialize the solution
#         sol = None
        
#         # Try different initializations
#         for _ in range(10):
#             # Randomly initialize the solution
#             sol = np.random.uniform(bounds, size=self.dim)
            
#             # Evaluate the function at the solution
#             func_sol = self.__call__(func, sol)
            
#             # Check if the solution is better than the current best
#             if func_sol < self.__call__(func, sol):
#                 # Update the solution
#                 sol = sol
        
#         # Return the best solution found
#         return sol

# def fitness_func(individual, func):
#     return func(individual)

# def mutate(individual):
#     idx = np.random.randint(0, individual.size)
#     individual[idx], individual[np.random.randint(0, individual.size)] = individual[np.random.randint(0, individual.size)], individual[idx]
#     return individual

# def evolve(population):
#     new_population = deque()
    
#     for _ in range(len(population) // 2):
#         parent1 = np.random.choice(population, size=individual.size, replace=False)
#         parent2 = np.random.choice(population, size=individual.size, replace=False)
#         child = (parent1 + parent2) / 2
        
#         child = np.clip(child, -5.0, 5.0)
        
#         new_population.append(child)
    
#     for individual in new_population:
#         if np.random.rand() < 0.25:
#             mutate(individual)
    
#     return list(new_population)

# # Test the algorithm
# budget = 100
# dim = 10
# func = np.sin
# population = [np.random.uniform(-10, 10, dim) for _ in range(100)]
# algorithm = BBOBMetaheuristic(budget, dim)
# fitness_scores = algorithm.evaluate_fitness(func, population)
# print(fitness_scores)