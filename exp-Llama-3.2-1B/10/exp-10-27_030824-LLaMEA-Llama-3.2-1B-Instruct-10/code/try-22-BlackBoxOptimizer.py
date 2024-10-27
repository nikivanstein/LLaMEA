# Description: Novel Metaheuristic Algorithm for Black Box Optimization
# Code: 
# ```python
import random
import numpy as np

class BlackBoxOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.search_space = [-5.0, 5.0]
        self.func_evaluations = 0
        self.new_individuals = []
        self.best_individual = None
        self.best_fitness = float('-inf')

    def __call__(self, func):
        # Ensure the function evaluations do not exceed the budget
        if self.func_evaluations < self.budget:
            # Generate a random point in the search space
            point = np.random.uniform(self.search_space[0], self.search_space[1])
            # Evaluate the function at the point
            evaluation = func(point)
            # Increment the function evaluations
            self.func_evaluations += 1
            # Return the point and its evaluation
            return point, evaluation
        else:
            # If the budget is reached, return a default point and evaluation
            return np.random.uniform(self.search_space[0], self.search_space[1]), func(np.random.uniform(self.search_space[0], self.search_space[1]))

    def mutate(self, individual):
        # Randomly swap two elements in the individual
        i, j = random.sample(range(len(individual)), 2)
        individual[i], individual[j] = individual[j], individual[i]
        # Ensure the individual is within the search space
        individual[i] = np.clip(individual[i], self.search_space[0], self.search_space[1])
        individual[j] = np.clip(individual[j], self.search_space[0], self.search_space[1])

    def evaluate_fitness(self, individual):
        # Evaluate the function at the individual
        evaluation = func(individual)
        # Return the fitness
        return evaluation

    def update_individual(self, individual, fitness):
        # If the individual has a better fitness than the best individual found so far
        if fitness > self.best_fitness:
            # Update the best individual and its fitness
            self.best_individual = individual
            self.best_fitness = fitness

    def __str__(self):
        # Return a string representation of the algorithm
        return f"Novel Metaheuristic Algorithm for Black Box Optimization"

def func(individual, dim):
    # Define the black box function
    return individual[dim] ** 2

# Initialize the algorithm
optimizer = BlackBoxOptimizer(1000, 10)

# Generate a random initial population
for _ in range(100):
    optimizer.new_individuals.append([random.uniform(-5.0, 5.0) for _ in range(dim)])

# Evaluate the fitness of the initial population
for individual in optimizer.new_individuals:
    fitness = optimizer.evaluate_fitness(individual)
    optimizer.update_individual(individual, fitness)

# Run the algorithm
for _ in range(100):
    # Generate a new individual
    individual = [random.uniform(-5.0, 5.0) for _ in range(dim)]
    # Evaluate the fitness of the new individual
    fitness = optimizer.evaluate_fitness(individual)
    # If the fitness is better than the best fitness found so far, update the best individual
    if fitness > optimizer.best_fitness:
        optimizer.update_individual(individual, fitness)
    # If the budget is reached, return a default individual and evaluation
    if optimizer.func_evaluations == optimizer.budget:
        individual = np.random.uniform(-5.0, 5.0), func(np.random.uniform(-5.0, 5.0))
        optimizer.update_individual(individual, func(individual))