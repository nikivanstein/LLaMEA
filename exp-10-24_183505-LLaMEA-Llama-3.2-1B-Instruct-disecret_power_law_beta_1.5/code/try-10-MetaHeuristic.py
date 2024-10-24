# Description: Novel Metaheuristic Algorithm for Black Box Optimization
# Code: 
# ```python
import random
import numpy as np
import matplotlib.pyplot as plt

class MetaHeuristic:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.search_space = [-5.0, 5.0]  # Define the search space
        self.best_func = None  # Initialize the best function found so far
        self.best_fitness = float('inf')  # Initialize the best fitness found so far
        self.iterations = 0  # Initialize the number of iterations

    def __call__(self, func, max_evals):
        # Evaluate the function up to max_evals times
        for _ in range(max_evals):
            # Randomly sample a point in the search space
            point = (random.uniform(self.search_space[0], self.search_space[1]), random.uniform(self.search_space[0], self.search_space[1]))
            # Evaluate the function at the point
            fitness = func(point)
            # If the fitness is better than the current best fitness, update the best function and fitness
            if fitness < self.best_fitness:
                self.best_func = func
                self.best_fitness = fitness
            # If the fitness is equal to the current best fitness, update the best function if it has a lower budget
            elif fitness == self.best_fitness and self.budget < self.best_func.budget:
                self.best_func = func
                self.best_fitness = fitness
        # Return the best function found
        return self.best_func

    def set_budget(self, budget):
        self.budget = budget

    def get_best_func(self):
        return self.best_func

def get_best_individual(meta_heuristic, func, max_evals):
    # Initialize the best individual and fitness
    best_individual = None
    best_fitness = float('inf')

    # Evaluate the function for the best individual
    for _ in range(max_evals):
        # Randomly sample a point in the search space
        point = (random.uniform(-5.0, 5.0), random.uniform(-5.0, 5.0))
        # Evaluate the function at the point
        fitness = func(point)
        # If the fitness is better than the current best fitness, update the best individual and fitness
        if fitness < best_fitness:
            best_individual = point
            best_fitness = fitness

    # Return the best individual
    return best_individual

def mutate(individual, meta_heuristic):
    # Randomly mutate the individual
    mutated_individual = (individual[0] + random.uniform(-1.0, 1.0), individual[1] + random.uniform(-1.0, 1.0))
    # Return the mutated individual
    return mutated_individual

# Define a noiseless function
def noiseless_func(x):
    return np.sin(x)

# Define a noise function
def noise_func(x):
    return np.random.normal(0, 1, x)

# Define a test function
def test_func(x):
    return x**2 + 2*x + 1

# Create an instance of the MetaHeuristic class
meta_heuristic = MetaHeuristic(100, 10)

# Set the budget for the MetaHeuristic
meta_heuristic.set_budget(100)

# Optimize the test function using the MetaHeuristic
best_func = meta_heuristic(__call__, 100)

# Print the best function found
print("Best function:", best_func)
print("Best fitness:", best_func.budget)

# Refine the strategy using mutation
def mutate_strategy(meta_heuristic, func, max_evals, mutation_rate):
    # Initialize the best individual and fitness
    best_individual = None
    best_fitness = float('inf')

    # Evaluate the function for the best individual
    for _ in range(max_evals):
        # Randomly sample a point in the search space
        point = (random.uniform(-5.0, 5.0), random.uniform(-5.0, 5.0))
        # Evaluate the function at the point
        fitness = func(point)
        # If the fitness is better than the current best fitness, update the best individual and fitness
        if fitness < best_fitness:
            best_individual = point
            best_fitness = fitness

    # Mutate the best individual
    mutated_individual = mutate(best_individual, meta_heuristic)
    # Evaluate the function for the mutated individual
    fitness = func(mutated_individual)
    # If the fitness is better than the current best fitness, update the best individual and fitness
    if fitness < best_fitness:
        best_individual = mutated_individual
        best_fitness = fitness

    # Return the best individual
    return best_individual

# Refine the strategy using mutation with 10% mutation rate
best_func = mutate_strategy(meta_heuristic, test_func, 100, 0.1)

# Print the best function found
print("Refined best function:", best_func)
print("Refined best fitness:", best_func.budget)