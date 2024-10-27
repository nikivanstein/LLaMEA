# Description: A novel metaheuristic algorithm for solving black box optimization problems.
# Code: 
# ```python
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

    def mutate(self, x):
        # Randomly mutate the individual by adding or subtracting a small value from each dimension
        return x + np.random.uniform(-1, 1, self.dim)

# One-line description with the main idea
# A novel metaheuristic algorithm for solving black box optimization problems using gradient descent with mutation.

class BBOBOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.explore_rate = 0.1
        self.learning_rate = 0.01
        self.explore_count = 0
        self.max_explore_count = budget

    def __call__(self, func, initial_individual, population_size):
        # Initialize the population with the initial individual
        population = [initial_individual] * population_size
        
        # Evolve the population over many generations
        for _ in range(1000):  # Limit the number of generations to 1000
            for individual in population:
                # Evaluate the function at the current individual
                y = inner(individual)
                
                # If we've reached the maximum number of iterations, stop evolving
                if self.explore_count >= self.max_explore_count:
                    break
                
                # If we've reached the upper bound, stop evolving
                if individual[-1] >= 5.0:
                    break
                
                # Learn a new direction using gradient descent
                learning_rate = self.learning_rate * (1 - self.explore_rate / self.max_explore_count)
                dx = -np.dot(individual - inner(individual), np.gradient(y))
                individual += learning_rate * dx
                
                # Update the exploration count
                self.explore_count += 1
                
                # If we've reached the upper bound, stop evolving
                if individual[-1] >= 5.0:
                    break
        
        # Select the fittest individual to reproduce
        fittest_individual = population[np.argmax([inner(individual) for individual in population])]
        
        # Create a new offspring by mutating the fittest individual
        offspring = [mutate(individual) for individual in population]
        
        # Replace the worst individual with the new offspring
        population[population.index(fittest_individual)] = offspring
        
        # Return the fittest individual
        return fittest_individual

def inner(func, x):
    return func(x)

# Test the algorithm
optimizer = MGDALR(100, 5)
optimizer.__call__(inner, np.array([-5.0] * 5), 100)