import numpy as np
import random

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

def mutation(individual, bounds):
    # Randomly mutate the individual
    mutated_individual = individual.copy()
    for i in range(len(individual)):
        if random.random() < 0.25:
            mutated_individual[i] += random.uniform(-1, 1)
            if mutated_individual[i] < -bounds[i]:
                mutated_individual[i] = -bounds[i]
            elif mutated_individual[i] > bounds[i]:
                mutated_individual[i] = bounds[i]
    return mutated_individual

def selection(population, bounds, num_pop):
    # Select the fittest individuals
    fitnesses = [self.__call__(func, individual) for func, individual in zip(population, population)]
    sorted_indices = np.argsort(fitnesses)
    sorted_indices = sorted_indices[:num_pop]
    selected_individuals = [population[i] for i in sorted_indices]
    return selected_individuals

def crossover(parent1, parent2, bounds):
    # Perform crossover
    child1 = np.concatenate((parent1[:len(parent1)//2], parent2[len(parent2)//2:]))
    child2 = np.concatenate((parent2[:len(parent2)//2], parent1[len(parent1)//2:]))
    return child1, child2

def bbobmetaheuristic(self, func, num_evals):
    # Initialize the population
    population = [func(np.random.uniform(bounds, size=self.dim)) for _ in range(num_evals)]
    
    # Perform selection, crossover, and mutation
    for _ in range(100):
        # Select the fittest individuals
        selected_individuals = selection(population, bounds, num_evals)
        
        # Perform crossover and mutation
        new_individuals = []
        for i in range(len(selected_individuals)):
            parent1, parent2 = selected_individuals[i], selected_individuals[(i+1) % len(selected_individuals)]
            child1, child2 = crossover(parent1, parent2, bounds)
            new_individuals.extend([child1, child2])
        
        # Update the population
        population = new_individuals
    
    # Return the best solution found
    return max(population)

# One-line description with the main idea
# BBOBMetaheuristic: A novel metaheuristic algorithm that uses genetic programming to optimize black box functions by refining its strategy through mutation, selection, and crossover.

# Code:
# ```python
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

# def mutation(individual, bounds):
#     # Randomly mutate the individual
#     mutated_individual = individual.copy()
#     for i in range(len(individual)):
#         if random.random() < 0.25:
#             mutated_individual[i] += random.uniform(-1, 1)
#             if mutated_individual[i] < -bounds[i]:
#                 mutated_individual[i] = -bounds[i]
#             elif mutated_individual[i] > bounds[i]:
#                 mutated_individual[i] = bounds[i]
    
#     return mutated_individual

# def selection(population, bounds, num_pop):
#     # Select the fittest individuals
#     fitnesses = [self.__call__(func, individual) for func, individual in zip(population, population)]
#     sorted_indices = np.argsort(fitnesses)
#     sorted_indices = sorted_indices[:num_pop]
#     selected_individuals = [population[i] for i in sorted_indices]
    
#     return selected_individuals

# def crossover(parent1, parent2, bounds):
#     # Perform crossover
#     child1 = np.concatenate((parent1[:len(parent1)//2], parent2[len(parent2)//2:]))
#     child2 = np.concatenate((parent2[:len(parent2)//2], parent1[len(parent1)//2:]))
    
#     return child1, child2

# def bbobmetaheuristic(self, func, num_evals):
#     # Initialize the population
#     population = [func(np.random.uniform(bounds, size=self.dim)) for _ in range(num_evals)]
    
#     # Perform selection, crossover, and mutation
#     for _ in range(100):
#         # Select the fittest individuals
#         selected_individuals = selection(population, bounds, num_evals)
        
#         # Perform crossover and mutation
#         new_individuals = []
#         for i in range(len(selected_individuals)):
#             parent1, parent2 = selected_individuals[i], selected_individuals[(i+1) % len(selected_individuals)]
#             child1, child2 = crossover(parent1, parent2, bounds)
#             new_individuals.extend([child1, child2])
        
#         # Update the population
#         population = new_individuals
    
#     # Return the best solution found
#     return max(population)

# # One-line description with the main idea
# # BBOBMetaheuristic: A novel metaheuristic algorithm that uses genetic programming to optimize black box functions by refining its strategy through mutation, selection, and crossover.