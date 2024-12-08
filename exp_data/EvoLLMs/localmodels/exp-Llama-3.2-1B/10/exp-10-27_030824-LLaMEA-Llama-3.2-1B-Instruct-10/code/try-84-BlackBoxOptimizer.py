import random
import numpy as np
from scipy.optimize import minimize

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

    def novel_metaheuristic(self, func, initial_point, iterations=100):
        # Define the mutation and crossover operators
        def mutate(individual):
            # Randomly change two genes in the individual
            mutated_individual = individual.copy()
            mutated_individual[1] = random.uniform(-5.0, 5.0)
            mutated_individual[2] = random.uniform(-5.0, 5.0)
            return mutated_individual

        def crossover(parent1, parent2):
            # Perform a simple crossover with replacement
            child1 = parent1[:len(parent1)//2] + [random.uniform(-5.0, 5.0) for _ in range(len(parent1)//2)]
            child2 = parent2[:len(parent2)//2] + [random.uniform(-5.0, 5.0) for _ in range(len(parent2)//2)]
            return child1, child2

        # Initialize the population
        population = [initial_point] * self.dim

        # Evolve the population for the specified number of iterations
        for _ in range(iterations):
            # Evaluate the fitness of each individual in the population
            fitnesses = [func(individual) for individual in population]

            # Select the fittest individuals to reproduce
            fittest_individuals = np.argsort(fitnesses)[-self.budget:]

            # Create a new population by crossover and mutation
            new_population = []
            for _ in range(self.dim):
                parent1 = population[fittest_individuals[_]]
                parent2 = population[fittest_individuals[_ + self.dim]]
                child = crossover(parent1, parent2)
                new_population.append(mutate(child))

            # Replace the old population with the new one
            population = new_population

        # Evaluate the fitness of the final population
        fitnesses = [func(individual) for individual in population]
        best_individual = np.argsort(fitnesses)[-1]

        # Return the best individual and its fitness
        return best_individual, fitnesses[-1]

# One-line description: Novel metaheuristic algorithm for black box optimization using a novel combination of random walk and linear interpolation.

# Code: 