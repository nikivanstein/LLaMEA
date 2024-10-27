import numpy as np
import random

class MGDALR:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.explore_rate = 0.1
        self.learning_rate = 0.01
        self.explore_count = 0
        self.max_explore_count = budget
        self.population_size = 100
        self.mutation_rate = 0.01
        self.population = []

    def __call__(self, func):
        def inner(x):
            return func(x)
        
        # Initialize x to the lower bound
        x = np.array([-5.0] * self.dim)
        
        for _ in range(self.budget):
            # Evaluate the function at the current x
            y = inner(x)
            
            # If we've reached the maximum number of iterations, stop exploring
            if self.explore_count >= self.max_explore_count:
                break
            
            # If we've reached the upper bound, stop exploring
            if x[-1] >= 5.0:
                break
            
            # Learn a new direction using gradient descent
            learning_rate = self.learning_rate * (1 - self.explore_rate / self.max_explore_count)
            dx = -np.dot(x - inner(x), np.gradient(y))
            x += learning_rate * dx
            
            # Update the exploration count
            self.explore_count += 1
            
            # If we've reached the upper bound, stop exploring
            if x[-1] >= 5.0:
                break
        
        # Refine the strategy using evolutionary algorithms
        self.refine_strategy(x)
        
        return x

    def refine_strategy(self, x):
        # Define a simple genetic algorithm to refine the strategy
        self.population = self.population[:self.population_size]
        for _ in range(100):
            # Select parents using tournament selection
            parents = random.sample(self.population, self.population_size)
            
            # Crossover (recombine) parents to create offspring
            offspring = []
            for _ in range(self.population_size):
                parent1, parent2 = random.sample(parents, 2)
                child = (0.5 * (parent1 + parent2)) + (0.5 * (parent1 - parent2))
                offspring.append(child)
            
            # Mutate offspring to introduce genetic variation
            for i in range(self.population_size):
                if random.random() < self.mutation_rate:
                    offspring[i] += random.uniform(-1, 1)
            
            # Replace the old population with the new one
            self.population = offspring

# Description: 
# A novel black box optimization algorithm that combines adaptive gradient descent with evolutionary strategies
# 
# Code: 
# ```python
# MGDALR
# ```
# ```python
# def __init__(self, budget, dim):
#     self.budget = budget
#     self.dim = dim
#     self.explore_rate = 0.1
#     self.learning_rate = 0.01
#     self.explore_count = 0
#     self.max_explore_count = budget
#     self.population_size = 100
#     self.mutation_rate = 0.01
#     self.population = []
# 
# def __call__(self, func):
#     def inner(x):
#         return func(x)
# 
#     # Initialize x to the lower bound
#     x = np.array([-5.0] * self.dim)
# 
#     for _ in range(self.budget):
#         # Evaluate the function at the current x
#         y = inner(x)
# 
#         # If we've reached the maximum number of iterations, stop exploring
#         if self.explore_count >= self.max_explore_count:
#             break
# 
#         # If we've reached the upper bound, stop exploring
#         if x[-1] >= 5.0:
#             break
# 
#         # Learn a new direction using gradient descent
#         learning_rate = self.learning_rate * (1 - self.explore_rate / self.max_explore_count)
#         dx = -np.dot(x - inner(x), np.gradient(y))
#         x += learning_rate * dx
# 
#         # Update the exploration count
#         self.explore_count += 1
# 
#     # Refine the strategy using evolutionary algorithms
#     self.refine_strategy(x)
# 
# def refine_strategy(self, x):
#     # Define a simple genetic algorithm to refine the strategy
#     self.population = self.population[:self.population_size]
#     for _ in range(100):
#         # Select parents using tournament selection
#         parents = random.sample(self.population, self.population_size)
# 
#         # Crossover (recombine) parents to create offspring
#         offspring = []
#         for _ in range(self.population_size):
#             parent1, parent2 = random.sample(parents, 2)
#             child = (0.5 * (parent1 + parent2)) + (0.5 * (parent1 - parent2))
#             offspring.append(child)
# 
#         # Mutate offspring to introduce genetic variation
#         for i in range(self.population_size):
#             if random.random() < self.mutation_rate:
#                 offspring[i] += random.uniform(-1, 1)
# 
#         # Replace the old population with the new one
#         self.population = offspring