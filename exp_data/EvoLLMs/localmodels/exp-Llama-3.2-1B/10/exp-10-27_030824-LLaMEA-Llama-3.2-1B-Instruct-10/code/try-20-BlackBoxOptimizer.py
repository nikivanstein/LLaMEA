import random
import numpy as np
import copy
from scipy.optimize import minimize_scalar

class BlackBoxOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.search_space = [-5.0, 5.0]
        self.func_evaluations = 0

    def __call__(self, func, initial_point, initial_fitness, iterations):
        # Initialize the current point, fitness, and population
        current_point = initial_point
        current_fitness = initial_fitness
        population = [copy.deepcopy(current_point) for _ in range(iterations)]

        # Run the optimization algorithm
        for _ in range(iterations):
            # Generate a random point in the search space
            point = np.random.uniform(self.search_space[0], self.search_space[1])
            # Evaluate the function at the point
            evaluation = func(point)

            # Update the current point, fitness, and population
            current_point, current_fitness = point, evaluation
            population[0], population[1] = point, evaluation

            # Evaluate the fitness of the population
            fitnesses = [func(point) for point in population]

            # Select the fittest individuals
            fittest_individuals = sorted(population, key=lambda x: x[1], reverse=True)[:self.budget]

            # Create a new population by combining the fittest individuals
            new_population = []
            for _ in range(self.budget):
                new_individual = fittest_individuals[0]
                new_individual[1] = fittest_individuals[0][1] + random.uniform(-0.1, 0.1)
                new_population.append(new_individual)

            # Replace the current population with the new population
            population = new_population

            # Update the current point, fitness, and population
            current_point, current_fitness = point, evaluation

        # Return the final population
        return population

# One-line description: Novel metaheuristic algorithm for black box optimization using a combination of random walk and linear interpolation.