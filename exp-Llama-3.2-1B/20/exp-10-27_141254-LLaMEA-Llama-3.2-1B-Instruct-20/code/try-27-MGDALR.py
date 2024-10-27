# Description: Novel Metaheuristic Algorithm for Black Box Optimization
# Code: 
# ```python
import numpy as np
import random
import math

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

def evaluate_fitness(individual, func, budget, logger):
    # Evaluate the fitness of the individual using the given function
    fitness = func(individual)
    logger.update_fitness(individual, fitness)
    return fitness

def generate_individual(dim):
    # Generate a random individual within the search space
    return np.random.rand(dim)

def mutate(individual):
    # Randomly mutate the individual within the search space
    mutated_individual = individual + np.random.rand(individual.shape[0]) * 0.1
    return mutated_individual

def create_individual(budget):
    # Create a new individual with a random fitness value
    individual = generate_individual(budget)
    fitness = evaluate_fitness(individual, func, budget, logger)
    return individual, fitness

def create_individual_with_strategy(budget):
    # Create a new individual with a random fitness value
    individual = generate_individual(budget)
    # Refine the strategy using gradient descent
    learning_rate = 0.01
    dx = -np.dot(individual - evaluate_fitness(individual, func, budget, logger), np.gradient(evaluate_fitness(individual, func, budget, logger)))
    individual += learning_rate * dx
    return individual

# Test the algorithm
budget = 100
dim = 10
func = np.random.rand(dim)
logger = MGDALR(budget, dim)

individual, fitness = create_individual(budget)
print(f"Initial individual: {individual}, Fitness: {fitness}")

# Refine the strategy
individual, fitness = create_individual_with_strategy(budget)
print(f"Refined individual: {individual}, Fitness: {fitness}")

# Refine the strategy again
individual, fitness = create_individual_with_strategy(budget)
print(f"Refined individual: {individual}, Fitness: {fitness}")