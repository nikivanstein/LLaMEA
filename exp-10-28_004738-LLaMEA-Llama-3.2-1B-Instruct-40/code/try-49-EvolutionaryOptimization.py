import random
import numpy as np
from scipy.optimize import minimize

class EvolutionaryOptimization:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.funcs = self.generate_functions()

    def generate_functions(self):
        functions = []
        for _ in range(24):
            func = lambda x: random.uniform(-5.0, 5.0)
            functions.append(func)
        return functions

    def __call__(self, func, x0, bounds, mutation_prob, mutation_rate):
        # Initialize the population with random individuals
        population = [x0] * 100
        for _ in range(self.budget):
            # Evaluate the fitness of each individual
            fitnesses = [func(individual, *bounds) for individual, bounds in zip(population, bounds)]
            # Select the fittest individuals
            population = [individual for _, individual in sorted(zip(fitnesses, population), reverse=True)[:self.dim]]
            # Perform mutation on each individual
            for individual in population:
                if random.random() < mutation_prob:
                    # Randomly select a mutation point
                    mutation_point = random.randint(0, self.dim - 1)
                    # Perform mutation
                    individual[mutation_point] += np.random.uniform(-1, 1) / self.dim
                    # Ensure the mutation point is within the bounds
                    individual[mutation_point] = max(bounds[0], min(bounds[1], individual[mutation_point]))
        # Return the fittest individual
        return population[0]

    def select_best(self, population):
        # Evaluate the fitness of each individual
        fitnesses = [func(individual, *bounds) for individual, bounds in zip(population, self.bounds)]
        # Return the fittest individual
        return population[np.argmax(fitnesses)]

    def mutate(self, individual, mutation_prob, mutation_rate):
        # Randomly select a mutation point
        mutation_point = random.randint(0, self.dim - 1)
        # Perform mutation
        individual[mutation_point] += np.random.uniform(-1, 1) / self.dim
        # Ensure the mutation point is within the bounds
        individual[mutation_point] = max(self.bounds[0], min(self.bounds[1], individual[mutation_point]))

# Description: Evolutionary Optimization using Hyper-Heuristics
# Code: 