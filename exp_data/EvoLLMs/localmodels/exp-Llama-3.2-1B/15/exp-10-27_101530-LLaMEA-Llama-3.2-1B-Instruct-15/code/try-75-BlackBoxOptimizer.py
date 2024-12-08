import random
import numpy as np

class BlackBoxOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.search_space = (-5.0, 5.0)
        self.func_evaluations = 0
        self.current_exp = 0
        self.best_individual = None
        self.best_fitness = float('inf')

    def __call__(self, func):
        while self.func_evaluations < self.budget:
            # Generate a random point in the search space
            point = (random.uniform(self.search_space[0], self.search_space[1]), random.uniform(self.search_space[0], self.search_space[1]))
            # Evaluate the function at the point
            func_value = func(point)
            # Increment the function evaluations
            self.func_evaluations += 1
            # Check if the point is within the budget
            if self.func_evaluations < self.budget:
                # If not, return the point
                return point
        # If the budget is reached, return the best point found so far
        return self.search_space[0], self.search_space[1]

    def mutate(self, individual):
        # Refine the strategy by changing a single line of code
        if random.random() < 0.15:
            # Change the individual line of code
            individual[0] = random.uniform(-5.0, 5.0)
            # Update the best individual
            self.best_individual = individual
            # Update the best fitness
            self.best_fitness = func(individual)
        # Return the mutated individual
        return individual

    def evaluate_fitness(self, individual):
        # Evaluate the fitness of the individual
        func_value = func(individual)
        # Update the best fitness
        self.best_fitness = func_value
        # Update the best individual
        self.best_individual = individual
        # Return the fitness
        return func_value

def func(individual):
    # Evaluate the black box function
    return individual[0] ** 2 + individual[1] ** 2

# Initialize the BlackBoxOptimizer with a budget and dimension
optimizer = BlackBoxOptimizer(100, 10)

# Optimize the function using the BlackBoxOptimizer
individual = optimizer(BlackBoxOptimizer(100, 10))
print(individual)

# Evaluate the fitness of the individual
fitness = optimizer.func(individual)
print(fitness)