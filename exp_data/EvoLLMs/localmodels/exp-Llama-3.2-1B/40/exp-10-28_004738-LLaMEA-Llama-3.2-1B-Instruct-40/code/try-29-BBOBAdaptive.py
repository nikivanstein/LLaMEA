import random
import numpy as np
from scipy.optimize import minimize

class BBOBAdaptive:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.funcs = self.generate_functions()
        self.population = self.initialize_population()

    def generate_functions(self):
        functions = []
        for _ in range(24):
            func = lambda x: random.uniform(-5.0, 5.0)
            functions.append(func)
        return functions

    def initialize_population(self):
        population = []
        for _ in range(100):
            individual = np.random.uniform(-5.0, 5.0, self.dim)
            population.append(individual)
        return population

    def __call__(self, func, bounds, mutation_prob, evolution_prob, budget):
        # Initialize the population with random individuals
        self.population = self.initialize_population()

        for _ in range(budget):
            # Evaluate the fitness of each individual
            fitnesses = [func(individual, bounds, mutation_prob, evolution_prob) for individual in self.population]

            # Select the fittest individuals for mutation
            fittest_individuals = self.population[np.argsort(fitnesses)[::-1][:self.budget]]

            # Perform mutation on the fittest individuals
            mutated_individuals = []
            for individual in fittest_individuals:
                mutated_individual = individual + np.random.uniform(-1, 1, self.dim)
                mutated_individuals.append(mutated_individual)

            # Select the new population with mutation
            new_population = []
            for _ in range(self.budget):
                # Evaluate the fitness of each individual
                fitnesses = [func(individual, bounds, mutation_prob, evolution_prob) for individual in mutated_individuals]
                # Select the fittest individual
                fittest_individual = mutated_individuals[np.argsort(fitnesses)[::-1][0]]
                # Add the fittest individual to the new population
                new_population.append(fittest_individual)

            # Update the population with the new population
            self.population = new_population

            # Update the bounds with the new bounds
            for individual in mutated_individuals:
                bounds = [bounds[0] - 1, bounds[1] + 1]

            # Update the bounds with the new bounds
            for individual in new_population:
                bounds = [bounds[0] - 1, bounds[1] + 1]

        # Return the best individual
        return self.population[np.argmin(fitnesses)]

    def f(self, x):
        return x**2 + 0.5*x + 0.1

    def f_prime(self, x):
        return 2*x + 0.5

    def f_double_prime(self, x):
        return 2

    def f_double_prime_prime(self, x):
        return 4

    def bbo_opt(self, func, bounds, mutation_prob, evolution_prob, budget):
        return self.__call__(func, bounds, mutation_prob, evolution_prob, budget)

# Description: Black Box Optimization using BBOB
# Code: 
# ```python
# Black Box Optimization using BBOB
# Code: 
# ```