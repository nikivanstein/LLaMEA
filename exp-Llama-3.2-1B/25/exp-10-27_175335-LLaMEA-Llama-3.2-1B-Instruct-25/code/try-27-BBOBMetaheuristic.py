import numpy as np
from collections import deque

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

    def mutate(self, sol):
        # Refine the solution strategy by changing lines of the selected solution
        sol[0] = np.random.uniform(-5.0, 5.0)  # Lower bound
        sol[1] = np.random.uniform(-5.0, 5.0)  # Upper bound
        return sol

    def evolve(self, population_size, mutation_rate):
        # Evolve the population using the selection, crossover, and mutation operators
        population = deque([self.search(func) for func in np.random.choice(list(self.func_evals), population_size, replace=False)])
        
        for _ in range(10):  # Evolve for 10 generations
            # Select the fittest individuals
            fittest = deque([func for func, _ in population if self.__call__(func, self.search(func))])
            
            # Crossover the fittest individuals to create new offspring
            offspring = []
            while len(offspring) < population_size:
                parent1, parent2 = np.random.choice(fittest, 2, replace=False)
                child = self.mutate(np.random.uniform(parent1, parent2))
                offspring.append(child)
            
            # Mutate the offspring to introduce genetic variation
            offspring = np.array(offspring)
            offspring = self.mutate(offspring)
            
            # Replace the least fit individuals with the new offspring
            population = deque([func for func, _ in population if self.__call__(func, self.search(func))])
            population.extend(offspring)
        
        # Return the best individual found
        return population[0]

# Description: Evolutionary Algorithm for Black Box Optimization using Genetic Programming
# Code: 
# ```python
# BBOBMetaheuristic: Evolutionary Algorithm for Black Box Optimization using Genetic Programming
# 
# class BBOBMetaheuristic:
#     def __init__(self, budget, dim):
#         self.budget = budget
#         self.dim = dim
#         self.func_evals = 0

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

#     def mutate(self, sol):
#         # Refine the solution strategy by changing lines of the selected solution
#         sol[0] = np.random.uniform(-5.0, 5.0)  # Lower bound
#         sol[1] = np.random.uniform(-5.0, 5.0)  # Upper bound
#         return sol

#     def evolve(self, population_size, mutation_rate):
#         # Evolve the population using the selection, crossover, and mutation operators
#         population = deque([self.search(func) for func in np.random.choice(list(self.func_evals), population_size, replace=False)])
        
#         for _ in range(10):  # Evolve for 10 generations
#             # Select the fittest individuals
#             fittest = deque([func for func, _ in population if self.__call__(func, self.search(func))])
            
#             # Crossover the fittest individuals to create new offspring
#             offspring = []
#             while len(offspring) < population_size:
#                 parent1, parent2 = np.random.choice(fittest, 2, replace=False)
#                 child = self.mutate(np.random.uniform(parent1, parent2))
#                 offspring.append(child)
            
#             # Mutate the offspring to introduce genetic variation
#             offspring = np.array(offspring)
#             offspring = self.mutate(offspring)
            
#             # Replace the least fit individuals with the new offspring
#             population = deque([func for func, _ in population if self.__call__(func, self.search(func))])
#             population.extend(offspring)
        
#         # Return the best individual found
#         return population[0]

# BBOBMetaheuristic(budget=1000, dim=10)
# Mutations are performed using the mutate method.
# The evolutionary process is then run for 10 generations.
# The best individual found is returned.