import random
import numpy as np
from scipy.optimize import differential_evolution

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

    def novel_metaheuristic_algorithm(self, func, budget, dim):
        # Initialize the population with random solutions
        population = [np.random.uniform(self.search_space[0], self.search_space[1]) for _ in range(100)]

        # Evolve the population over multiple generations
        for _ in range(1000):
            # Evaluate the fitness of each individual in the population
            fitness = [func(individual) for individual in population]
            # Select the fittest individuals to reproduce
            fittest_individuals = np.argsort(fitness)[-self.budget:]

            # Create a new population by crossover and mutation
            new_population = []
            for _ in range(self.budget):
                # Select two parents from the fittest individuals
                parent1, parent2 = random.sample(fittest_individuals, 2)
                # Perform crossover to create a new individual
                child = np.clip(func(parent1) + func(parent2), self.search_space[0], self.search_space[1])
                # Perform mutation to introduce randomness
                child = np.clip(child + random.uniform(-1, 1), self.search_space[0], self.search_space[1])
                # Add the new individual to the new population
                new_population.append(child)

            # Replace the old population with the new population
            population = new_population

        # Evaluate the fitness of the final population
        fitness = [func(individual) for individual in population]
        # Return the fittest individual and its fitness
        return population[np.argmax(fitness)], fitness[np.argmax(fitness)]

# One-line description: Novel metaheuristic algorithm for black box optimization using a novel combination of random walk and linear interpolation.
