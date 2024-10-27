import random
import numpy as np

class BlackBoxOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.search_space = [-5.0, 5.0]
        self.func_evaluations = 0

    def __call__(self, func, iterations=1000):
        # Initialize the population with random points in the search space
        population = self.initialize_population(iterations)

        # Run the optimization algorithm for a specified number of iterations
        for _ in range(iterations):
            # Evaluate the fitness of each individual in the population
            fitnesses = [self.evaluate_fitness(individual) for individual in population]

            # Select the fittest individuals
            fittest_individuals = np.argsort(fitnesses)[-self.budget:]

            # Create a new population by linearly interpolating between the fittest individuals
            new_population = [self.interpolate(fittest_individuals, self.search_space) for _ in range(len(fittest_individuals))]

            # Replace the old population with the new one
            population = new_population

        # Evaluate the fitness of each individual in the final population
        fitnesses = [self.evaluate_fitness(individual) for individual in population]

        # Return the fittest individual and its fitness
        return self.get_fittest_individual(population, fitnesses), fitnesses

    def initialize_population(self, iterations):
        # Initialize the population with random points in the search space
        population = []
        for _ in range(iterations):
            point = np.random.uniform(self.search_space[0], self.search_space[1])
            population.append(point)
        return population

    def evaluate_fitness(self, individual):
        # Evaluate the fitness of an individual
        evaluation = self.func(individual)
        return evaluation

    def interpolate(self, fittest_individuals, search_space):
        # Interpolate between the fittest individuals
        new_individual = np.interp(fittest_individuals, fittest_individuals, search_space)
        return new_individual

    def get_fittest_individual(self, population, fitnesses):
        # Return the fittest individual
        return population[np.argmax(fitnesses)]

# One-line description: Novel metaheuristic algorithm for black box optimization using a novel combination of random walk and linear interpolation.
# Code: 