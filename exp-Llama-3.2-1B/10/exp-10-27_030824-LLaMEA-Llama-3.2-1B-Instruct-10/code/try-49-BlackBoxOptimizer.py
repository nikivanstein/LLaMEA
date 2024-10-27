import random
import numpy as np
from scipy.optimize import minimize
from collections import deque
import copy

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

    def novel_metaheuristic(self, func, population_size, mutation_rate, num_generations):
        # Initialize the population with random individuals
        population = [copy.deepcopy(func) for _ in range(population_size)]

        # Initialize the current generation
        current_generation = population.copy()

        # Initialize the best individual and its fitness
        best_individual = None
        best_fitness = float('-inf')

        # Iterate through the generations
        for _ in range(num_generations):
            # Evaluate the fitness of each individual
            fitnesses = [func(individual) for individual in current_generation]

            # Select the fittest individuals
            fittest_individuals = [individual for individual, fitness in zip(current_generation, fitnesses) if fitness == max(fitnesses)]

            # Mutate the fittest individuals
            mutated_individuals = []
            for _ in range(population_size):
                if random.random() < mutation_rate:
                    mutated_individuals.append(copy.deepcopy(fittest_individuals[_]))
                else:
                    mutated_individuals.append(fittest_individuals[_])

            # Replace the current generation with the mutated generation
            current_generation = mutated_individuals

            # Update the best individual and its fitness
            if len(fittest_individuals) > 0:
                best_individual = fittest_individuals[0]
                best_fitness = max(fitnesses)

        # Return the best individual and its fitness
        return best_individual, best_fitness

# One-line description: Novel metaheuristic algorithm for black box optimization using a novel combination of random walk and linear interpolation.
