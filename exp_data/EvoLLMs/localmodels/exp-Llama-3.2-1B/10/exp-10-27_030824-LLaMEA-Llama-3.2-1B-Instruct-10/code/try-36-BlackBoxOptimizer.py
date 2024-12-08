import numpy as np
import random

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

    def novel_metaheuristic(self, func, population_size, mutation_rate, max_iter):
        # Initialize the population
        population = [random.uniform(self.search_space[0], self.search_space[1]) for _ in range(population_size)]

        # Evaluate the function for each individual in the population
        for _ in range(max_iter):
            # Evaluate the function for each individual
            evaluations = [func(individual) for individual in population]
            # Calculate the fitness of each individual
            fitness = np.array([evaluations[i] / population_size for i in range(population_size)])
            # Select the fittest individuals
            fittest_individuals = population[np.argsort(fitness)[::-1][:population_size//2]]
            # Create a new generation
            new_population = [random.uniform(self.search_space[0], self.search_space[1]) for _ in range(population_size)]
            # Mutate the new generation
            for _ in range(population_size):
                parent1, parent2 = random.sample(fittest_individuals, 2)
                child = (parent1 + parent2) / 2
                if np.random.rand() < mutation_rate:
                    child += random.uniform(-self.search_space[0], self.search_space[1])
                new_population.append(child)
            # Replace the old population with the new population
            population = new_population

        # Evaluate the function for each individual in the new population
        new_evaluations = [func(individual) for individual in population]
        # Calculate the fitness of each individual in the new population
        fitness = np.array([new_evaluations[i] / population_size for i in range(population_size)])
        # Select the fittest individuals in the new population
        fittest_individuals = population[np.argsort(fitness)[::-1][:population_size//2]]
        # Return the fittest individual in the new population
        return fittest_individuals[0]

# One-line description: Novel metaheuristic algorithm for black box optimization using a novel combination of random walk and linear interpolation.

# Code: