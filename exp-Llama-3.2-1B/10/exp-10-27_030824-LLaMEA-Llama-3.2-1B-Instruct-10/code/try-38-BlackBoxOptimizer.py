import random
import numpy as np
from scipy.optimize import differential_evolution

class BlackBoxOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.search_space = [-5.0, 5.0]
        self.func_evaluations = 0

    def __call__(self, func, initial_individual, logger):
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

# One-line description: Novel metaheuristic algorithm for black box optimization using a novel combination of random walk and linear interpolation.

def random_walk(individual, budget):
    # Initialize the current point
    current_point = individual.copy()
    
    # Perform random walk for the specified budget
    for _ in range(budget):
        # Generate a new point using linear interpolation
        new_point = current_point.copy()
        for i in range(1, len(current_point)):
            new_point[i] = current_point[i] + random.uniform(-1, 1) / (i + 1)
        # Update the current point
        current_point = new_point
    
    # Evaluate the function at the final point
    evaluation = func(current_point)
    # Return the final point and evaluation
    return current_point, evaluation

def linear_interpolation(individual, budget):
    # Initialize the current point
    current_point = individual.copy()
    
    # Perform linear interpolation for the specified budget
    for _ in range(budget):
        # Generate a new point using linear interpolation
        new_point = current_point.copy()
        for i in range(1, len(current_point)):
            new_point[i] = current_point[i] + random.uniform(-1, 1) / (i + 1)
        # Update the current point
        current_point = new_point
    
    # Evaluate the function at the final point
    evaluation = func(current_point)
    # Return the final point and evaluation
    return current_point, evaluation

def differential_evolution(func, bounds, initial_individual, logger):
    # Initialize the population
    population = [initial_individual.copy() for _ in range(100)]
    
    # Run the differential evolution algorithm
    for _ in range(100):
        # Evaluate the fitness of each individual
        fitness = [func(individual) for individual in population]
        
        # Select the fittest individuals
        fittest_individuals = [individual for _, individual in sorted(zip(fitness, population), reverse=True)[:20]]
        
        # Create a new population by linearly interpolating the fittest individuals
        new_population = []
        for i in range(20):
            # Select two parents from the fittest individuals
            parent1 = fittest_individuals[i]
            parent2 = fittest_individuals[(i + 1) % 20]
            
            # Create a new child using linear interpolation
            child = linear_interpolation(parent1, 10)
            child = linear_interpolation(child, 10)
            
            # Add the child to the new population
            new_population.append(child)
        
        # Replace the old population with the new population
        population = new_population
    
    # Evaluate the fitness of the final population
    fitness = [func(individual) for individual in population]
    
    # Return the fittest individual and its fitness
    return population[fitness.index(max(fitness))], max(fitness)

# Initialize the BlackBoxOptimizer
optimizer = BlackBoxOptimizer(100, 5)
# Define the function to be optimized
def func(individual):
    return individual[0]**2 + individual[1]**2

# Run the differential evolution algorithm
best_individual, best_fitness = differential_evolution(func, [-10, 10], [optimizer.__call__(func, [0.0, 0.0], logger), optimizer.__call__(func, [0.0, 0.0]), optimizer.__call__(func, [0.0, 0.0])], logger)
# Print the result
print("Best individual:", best_individual)
print("Best fitness:", best_fitness)