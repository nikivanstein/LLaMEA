# Description: "Black Box Optimization using Metaheuristics"
# Code: 
# ```python
import random
import numpy as np

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

# Description: "Black Box Optimization using Genetic Algorithm"
# Code: 
# ```python
# import random
# import numpy as np
# import matplotlib.pyplot as plt

# Define the MetaHeuristic class
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

# Refine the strategy using genetic algorithm
def genetic_algorithm(func, max_evals, budget, dim):
    population = [func(random.uniform(-5, 5), random.uniform(-5, 5)) for _ in range(100)]
    while True:
        # Evaluate the fitness of each individual
        fitnesses = [func(individual, max_evals) for individual in population]
        # Select the fittest individuals
        fittest_individuals = [individual for _, individual in sorted(zip(fitnesses, population), reverse=True)[:10]]
        # Create a new population
        new_population = [func(random.uniform(-5, 5), random.uniform(-5, 5)) for individual in fittest_individuals]
        # Evaluate the fitness of each individual
        fitnesses = [func(individual, max_evals) for individual in new_population]
        # Select the fittest individuals
        fittest_individuals = [individual for _, individual in sorted(zip(fitnesses, new_population), reverse=True)[:10]]
        # Replace the old population with the new population
        population = fittest_individuals
        # Update the best function
        best_func = min(population, key=lambda individual: individual.budget)
        # Check if the best function has a lower budget
        if best_func.budget < meta_heuristic.get_best_func().budget:
            meta_heuristic.set_budget(best_func.budget)
            meta_heuristic.get_best_func = best_func
        # Check if the maximum number of iterations is reached
        if len(population) == 100:
            break
    return meta_heuristic.get_best_func()

# Optimize the test function using the genetic algorithm
best_func = genetic_algorithm(test_func, 100, 100, 10)

# Print the best function found
print("Best function:", best_func)
print("Best fitness:", best_func.budget)

# Refine the strategy using genetic algorithm (again)
best_func = genetic_algorithm(test_func, 100, 100, 10)

# Print the best function found
print("Best function:", best_func)
print("Best fitness:", best_func.budget)