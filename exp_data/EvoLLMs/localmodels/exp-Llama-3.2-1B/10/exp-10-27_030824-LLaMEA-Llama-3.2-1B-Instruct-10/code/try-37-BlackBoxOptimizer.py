import random
import numpy as np
from scipy.optimize import minimize_scalar

class BlackBoxOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.search_space = [-5.0, 5.0]
        self.func_evaluations = 0

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

    def novel_metaheuristic(self, func, initial_point, iterations=1000, mutation_rate=0.1):
        # Initialize the population with random points in the search space
        population = [initial_point + np.random.uniform(-1, 1, self.dim) for _ in range(100)]

        # Define the fitness function
        def fitness(individual):
            return func(individual)

        # Evaluate the fitness of each individual in the population
        for _ in range(iterations):
            # Select the fittest individuals to reproduce
            parents = population[np.argsort(fitness(population))[:50]]
            # Perform crossover to create offspring
            offspring = []
            for i in range(0, len(parents), 2):
                parent1, parent2 = parents[i], parents[i+1]
                child = (parent1 + parent2) / 2
                if random.random() < mutation_rate:
                    child += np.random.uniform(-1, 1, self.dim)
                offspring.append(child)
            # Evaluate the fitness of the offspring
            offspring_fitness = [fitness(individual) for individual in offspring]
            # Select the fittest offspring to reproduce
            parents = offspring[np.argsort(offspring_fitness)[:50]]
            # Replace the least fit individuals with the fittest offspring
            population = parents + offspring[:50]

        # Return the fittest individual in the final population
        return population[np.argmax(fitness(population))]

# One-line description: Novel metaheuristic algorithm for black box optimization using a novel combination of random walk and linear interpolation.

# Code: 