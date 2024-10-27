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

    def novel_metaheuristic(self, func, initial_individual, mutation_rate, num_generations):
        # Initialize the population with random individuals
        population = [initial_individual] * 100

        # Evaluate the fitness of each individual
        for _ in range(num_generations):
            # Select the fittest individuals
            fittest_individuals = sorted(population, key=lambda individual: individual.fun, reverse=True)[:self.budget]

            # Evaluate the fitness of the fittest individuals
            for individual in fittest_individuals:
                # Perform mutation
                mutated_individual = individual.copy()
                if random.random() < mutation_rate:
                    mutated_individual[0] = random.uniform(self.search_space[0], self.search_space[1])
                    mutated_individual[1] = func(mutated_individual[0])

                # Evaluate the fitness of the mutated individual
                new_fitness = individual.fun
                # Update the individual's fitness
                individual.fun = new_fitness

            # Replace the least fit individuals with the mutated ones
            population = [individual for individual in population if individual.fun >= fittest_individuals[0].fun] + \
                         [mutated_individual for mutated_individual in mutated_individuals if mutated_individual.fun >= fittest_individuals[0].fun]

# One-line description: Novel metaheuristic algorithm for black box optimization using a combination of random walk and linear interpolation.

# Code: 