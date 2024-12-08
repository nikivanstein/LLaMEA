import numpy as np
from typing import Dict

class BBOBMetaheuristic:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.func_evals = 0

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

    def mutate(self, func, sol):
        # Define the mutation rules
        mutation_rules = {
           'swap': lambda sol1, sol2: np.random.choice(sol1.shape),  # Swap two elements
            'insert': lambda sol1, sol2: np.concatenate((sol1, sol2)),  # Insert an element
            'delete': lambda sol1, sol2: np.delete(sol1, np.random.choice(sol1.shape)),  # Delete an element
           'scale': lambda sol1, sol2: sol1 * np.random.uniform(0.5, 2.0)  # Scale an element
        }
        
        # Apply the mutation rules
        for rule, func in mutation_rules.items():
            sol = func(sol, sol)
        
        # Check if the mutation was successful
        if np.any(np.abs(sol - self.__call__(func, sol)) > 0.01):
            raise ValueError("Mutation failed")
        
        # Return the mutated solution
        return sol

    def evolve(self, func, population_size):
        # Initialize the population
        population = np.random.uniform(bounds, size=(population_size, self.dim))
        
        # Evolve the population for a specified number of generations
        for _ in range(100):
            # Evaluate the function for each individual in the population
            func_evals = 0
            for _ in range(self.budget):
                func_evals += self.__call__(func, population)
            
            # Select the fittest individuals
            fittest_individuals = np.argsort(func_evals)[-population_size:]
            
            # Mutate the fittest individuals
            mutated_individuals = [self.mutate(func, individual) for individual in fittest_individuals]
            
            # Replace the least fit individuals with the mutated individuals
            population[fittest_individuals] = mutated_individuals
        
        # Return the best solution found
        return population[np.argmax(func_evals)]

class BBOBMetaheuristicEvolutionaryAlgorithm:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.func = None
        self.population_size = 100
    
    def __call__(self, func):
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

    def mutate(self, func, sol):
        # Define the mutation rules
        mutation_rules = {
           'swap': lambda sol1, sol2: np.random.choice(sol1.shape),  # Swap two elements
            'insert': lambda sol1, sol2: np.concatenate((sol1, sol2)),  # Insert an element
            'delete': lambda sol1, sol2: np.delete(sol1, np.random.choice(sol1.shape)),  # Delete an element
           'scale': lambda sol1, sol2: sol1 * np.random.uniform(0.5, 2.0)  # Scale an element
        }
        
        # Apply the mutation rules
        for rule, func in mutation_rules.items():
            sol = func(sol, sol)
        
        # Check if the mutation was successful
        if np.any(np.abs(sol - self.__call__(func, sol)) > 0.01):
            raise ValueError("Mutation failed")
        
        # Return the mutated solution
        return sol

    def evolve(self, func, population_size):
        # Initialize the population
        population = np.random.uniform(bounds, size=(population_size, self.dim))
        
        # Evolve the population for a specified number of generations
        for _ in range(100):
            # Evaluate the function for each individual in the population
            func_evals = 0
            for _ in range(self.budget):
                func_evals += self.__call__(func, population)
            
            # Select the fittest individuals
            fittest_individuals = np.argsort(func_evals)[-population_size:]
            
            # Mutate the fittest individuals
            mutated_individuals = [self.mutate(func, individual) for individual in fittest_individuals]
            
            # Replace the least fit individuals with the mutated individuals
            population[fittest_individuals] = mutated_individuals
        
        # Return the best solution found
        return population[np.argmax(func_evals)]

# Description: Evolutionary Algorithm for Black Box Optimization using Genetic Programming
# Code: 
# ```python
# import numpy as np
# import random
# import matplotlib.pyplot as plt

# class BBOBMetaheuristicEvolutionaryAlgorithm:
#     def __init__(self, budget, dim):
#         self.budget = budget
#         self.dim = dim
#         self.func = None
#         self.population_size = 100
    
#     def __call__(self, func):
#         # Define the search space
#         bounds = np.linspace(-5.0, 5.0, self.dim, endpoint=False)
#         
#         # Initialize the solution
#         sol = None
        
#         # Try different initializations
#         for _ in range(10):
#             # Randomly initialize the solution
#             sol = np.random.uniform(bounds, size=self.dim)
#             
#             # Evaluate the function at the solution
#             func_sol = self.__call__(func, sol)
#             
#             # Check if the solution is better than the current best
#             if func_sol < self.__call__(func, sol):
#                 # Update the solution
#                 sol = sol
        
#         # Return the best solution found
#         return sol

#     def mutate(self, func, sol):
#         # Define the mutation rules
#         mutation_rules = {
#            'swap': lambda sol1, sol2: np.random.choice(sol1.shape),  # Swap two elements
#             'insert': lambda sol1, sol2: np.concatenate((sol1, sol2)),  # Insert an element
#             'delete': lambda sol1, sol2: np.delete(sol1, np.random.choice(sol1.shape)),  # Delete an element
#            'scale': lambda sol1, sol2: sol1 * np.random.uniform(0.5, 2.0)  # Scale an element
#         }
        
#         # Apply the mutation rules
#         for rule, func in mutation_rules.items():
#             sol = func(sol, sol)
        
#         # Check if the mutation was successful
#         if np.any(np.abs(sol - self.__call__(func, sol)) > 0.01):
#             raise ValueError("Mutation failed")
        
#         # Return the mutated solution
#         return sol

#     def evolve(self, func, population_size):
#         # Initialize the population
#         population = np.random.uniform(bounds, size=(population_size, self.dim))
        
#         # Evolve the population for a specified number of generations
#         for _ in range(100):
#             # Evaluate the function for each individual in the population
#             func_evals = 0
#             for _ in range(self.budget):
#                 func_evals += self.__call__(func, population)
            
#             # Select the fittest individuals
#             fittest_individuals = np.argsort(func_evals)[-population_size:]
            
#             # Mutate the fittest individuals
#             mutated_individuals = [self.mutate(func, individual) for individual in fittest_individuals]
            
#             # Replace the least fit individuals with the mutated individuals
#             population[fittest_individuals] = mutated_individuals
        
#         # Return the best solution found
#         return population[np.argmax(func_evals)]

# # Example usage:
# algorithm = BBOBMetaheuristicEvolutionaryAlgorithm(budget=100, dim=10)
# func = lambda x: np.sin(x)
# best_solution = algorithm.evolve(func, 1000)
# print("Best solution:", best_solution)
# plt.scatter(best_solution[:, 0], best_solution[:, 1], c='r')
# plt.show()

# # Description: Evolutionary Algorithm for Black Box Optimization using Genetic Programming
# Code: 
# ```python
# import numpy as np
# import random

# class BBOBMetaheuristicEvolutionaryAlgorithm:
#     def __init__(self, budget, dim):
#         self.budget = budget
#         self.dim = dim
#         self.func = None
#         self.population_size = 100
    
#     def __call__(self, func):
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

#     def mutate(self, func, sol):
#         # Define the mutation rules
#         mutation_rules = {
#            'swap': lambda sol1, sol2: np.random.choice(sol1.shape),  # Swap two elements
#             'insert': lambda sol1, sol2: np.concatenate((sol1, sol2)),  # Insert an element
#             'delete': lambda sol1, sol2: np.delete(sol1, np.random.choice(sol1.shape)),  # Delete an element
#            'scale': lambda sol1, sol2: sol1 * np.random.uniform(0.5, 2.0)  # Scale an element
#         }
        
#         # Apply the mutation rules
#         for rule, func in mutation_rules.items():
#             sol = func(sol, sol)
        
#         # Check if the mutation was successful
#         if np.any(np.abs(sol - self.__call__(func, sol)) > 0.01):
#             raise ValueError("Mutation failed")
        
#         # Return the mutated solution
#         return sol

#     def evolve(self, func, population_size):
#         # Initialize the population
#         population = np.random.uniform(bounds, size=(population_size, self.dim))
        
#         # Evolve the population for a specified number of generations
#         for _ in range(100):
#             # Evaluate the function for each individual in the population
#             func_evals = 0
#             for _ in range(self.budget):
#                 func_evals += self.__call__(func, population)
            
#             # Select the fittest individuals
#             fittest_individuals = np.argsort(func_evals)[-population_size:]
            
#             # Mutate the fittest individuals
#             mutated_individuals = [self.mutate(func, individual) for individual in fittest_individuals]
            
#             # Replace the least fit individuals with the mutated individuals
#             population[fittest_individuals] = mutated_individuals
        
#         # Return the best solution found
#         return population[np.argmax(func_evals)]

# # Example usage:
# algorithm = BBOBMetaheuristicEvolutionaryAlgorithm(budget=100, dim=10)
# func = lambda x: np.sin(x)
# best_solution = algorithm.evolve(func, 1000)
# print("Best solution:", best_solution)