import numpy as np
import random
from scipy.optimize import minimize

class NovelMetaheuristicOptimizer:
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
        # Define mutation rules
        if random.random() < 0.15:
            # Randomly swap two elements in the individual
            idx1, idx2 = random.sample(range(self.dim), 2)
            individual[idx1], individual[idx2] = individual[idx2], individual[idx1]
        # Define mutation rules for bounds
        if random.random() < 0.1:
            # Randomly adjust the bounds
            self.search_space[0] += random.uniform(-1, 1)
            self.search_space[1] += random.uniform(-1, 1)
        return individual

    def evolve(self, population_size, generations):
        # Initialize the population
        population = [self.__call__(func) for func in self.funcs]
        for _ in range(generations):
            # Evaluate the fitness of each individual
            fitnesses = [func(individual) for individual, func in zip(population, self.funcs)]
            # Select the fittest individuals
            fittest = np.argsort(fitnesses)[-self.population_size:]
            # Create a new population by mutating the fittest individuals
            new_population = [self.mutate(individual) for individual in fittest]
            # Replace the old population with the new one
            population = new_population
        return population

# Define the functions to be optimized
def func1(individual):
    return individual[0]**2 + individual[1]**2

def func2(individual):
    return -individual[0]**2 - individual[1]**2

# Define the number of functions to be optimized
num_functions = 24

# Create a new optimizer with the specified parameters
optimizer = NovelMetaheuristicOptimizer(budget=100, dim=5)

# Evolve the optimizer
population = optimizer.evolve(population_size=100, generations=1000)

# Print the final fitness of the fittest individual
print("Final Fitness:", optimizer.funcs[-1])