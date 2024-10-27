import numpy as np
import random
import math

class AdaptiveBBOOPOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.search_space = np.linspace(-5.0, 5.0, dim)
        self.func_evaluations = 0
        self.population_size = 50
        self.mutation_rate = 0.01
        self.crossover_rate = 0.5

    def __call__(self, func):
        while self.func_evaluations < self.budget:
            # Select a random parent using genetic programming
            parent = self.select_parent(self.population_size)
            # Generate two new children using crossover
            child1 = self.crossover(parent, self.search_space)
            child2 = self.crossover(parent, self.search_space)
            # Evaluate the fitness of the new generation
            fitness1 = self.evaluate_fitness(child1)
            fitness2 = self.evaluate_fitness(child2)
            # Select the fittest child
            new_individual = self.select_fittest(child1, fitness1, fitness2)
            # Replace the old generation with the new one
            self.population = [new_individual]
            # Update the search space
            self.search_space = np.linspace(-5.0, 5.0, self.dim)
        return self.population[-1]

    def select_parent(self, population_size):
        # Select a random parent using genetic programming
        return random.choice(population_size)

    def crossover(self, parent, search_space):
        # Perform crossover between two parents
        child1 = parent[:self.dim//2]
        child2 = parent[self.dim//2:]
        child1[::2] = search_space
        child2[::2] = search_space
        return child1, child2

    def evaluate_fitness(self, individual):
        # Evaluate the fitness of an individual
        func_value = individual[-1]
        if np.isnan(func_value) or np.isinf(func_value):
            raise ValueError("Invalid function value")
        if func_value < 0 or func_value > 1:
            raise ValueError("Function value must be between 0 and 1")
        return func_value

    def select_fittest(self, child1, fitness1, fitness2):
        # Select the fittest child
        if fitness1 < fitness2:
            return child1
        else:
            return child2

# Initialize the optimizer
optimizer = AdaptiveBBOOPOptimizer(100, 10)

# Add the current solution to the population
optimizer.population.append((optimizer, (0, 0, 0)))

# Run the optimization algorithm
while True:
    # Select a new individual using genetic programming
    individual = optimizer.__call__(optimizer)
    # Add the new individual to the population
    optimizer.population.append((optimizer, individual))
    # Check if the budget has been reached
    if len(optimizer.population) > optimizer.budget:
        break

# Print the final solution
print("Final Solution:", optimizer.population[-1][1])