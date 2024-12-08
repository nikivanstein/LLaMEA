# Description: Novel Metaheuristic Algorithm for Black Box Optimization
# Code: 
# ```python
import numpy as np
from scipy.optimize import differential_evolution

class MGDALR:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.explore_rate = 0.1
        self.learning_rate = 0.01
        self.explore_count = 0
        self.max_explore_count = budget

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
        
        return x

    def optimize(self, func, initial_guess, budget):
        # Create an initial population of random individuals
        population = [initial_guess] * self.dim
        
        # Evaluate the fitness of each individual in the population
        fitness = np.array([func(individual) for individual in population])
        
        # Select the best individuals to reproduce
        selected_individuals = np.array([population[np.argsort(-fitness)][:self.budget]])
        
        # Create a new population by refining the selected individuals
        new_population = self.__call__(func, selected_individuals)
        
        # Evaluate the fitness of the new population
        fitness_new = np.array([func(individual) for individual in new_population])
        
        # Select the best individuals to replace the old population
        selected_individuals = np.array([new_population[np.argsort(-fitness_new)][:self.budget]])
        
        # Return the new population and its fitness
        return selected_individuals, fitness_new

# One-line description with the main idea
# Novel Metaheuristic Algorithm for Black Box Optimization
# 
# This algorithm uses differential evolution to optimize black box functions, 
# learning a new direction using gradient descent and refining the strategy 
# based on the probability of changing the direction.