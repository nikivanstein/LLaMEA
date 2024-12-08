import random
import numpy as np
from scipy.optimize import minimize

class BlackBoxOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.search_space = (-5.0, 5.0)
        self.func_evaluations = 0

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
        # Randomly select a dimension to mutate
        dim_to_mutate = random.randint(0, self.dim - 1)
        # Generate a new mutation point
        new_point = (individual[0] + random.uniform(-1, 1), individual[1] + random.uniform(-1, 1))
        # Check if the mutation point is within the budget
        if self.func_evaluations + 1 < self.budget:
            # If not, return the original point
            return individual
        # Return the mutated point
        return new_point

    def evaluate_fitness(self, individual):
        # Evaluate the function at the individual
        func_value = self.func(individual)
        # Check if the individual is within the budget
        if self.func_evaluations + 1 < self.budget:
            # If not, return the function value
            return func_value
        # Return the function value
        return func_value

def black_box_optimizer(budget, dim):
    # Initialize the optimizer
    optimizer = BlackBoxOptimizer(budget, dim)
    # Initialize the population
    population = [optimizer.evaluate_fitness([0, 0]) for _ in range(100)]
    # Initialize the best solution
    best_individual = population[0]
    # Initialize the best fitness
    best_fitness = population[0]
    # Run the optimization algorithm
    for _ in range(1000):
        # Generate a new individual
        individual = optimizer.mutate(population[-1])
        # Evaluate the fitness of the individual
        fitness = optimizer.evaluate_fitness(individual)
        # Check if the fitness is better than the best fitness
        if fitness < best_fitness:
            # Update the best individual and fitness
            best_individual = individual
            best_fitness = fitness
        # Add the individual to the population
        population.append(individual)
    # Return the best individual and fitness
    return best_individual, best_fitness

# Description: Novel Metaheuristic Algorithm for Black Box Optimization
# Code: 
# ```python
# import numpy as np
# import random
# from scipy.optimize import minimize

def optimize_bbb(budget, dim):
    best_individual, best_fitness = black_box_optimizer(budget, dim)
    # Refine the strategy using probability 0.15
    for _ in range(100):
        # Generate a new individual
        individual = best_individual.mutate()
        # Evaluate the fitness of the individual
        fitness = best_fitness
        # Check if the fitness is better than the best fitness
        if fitness < best_fitness:
            # Update the best individual and fitness
            best_individual = individual
            best_fitness = fitness
    # Return the best individual and fitness
    return best_individual, best_fitness

# Example usage:
budget = 1000
dim = 2
best_individual, best_fitness = optimize_bbb(budget, dim)
print(f"Best individual: {best_individual}, Best fitness: {best_fitness}")