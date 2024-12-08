import numpy as np
from scipy.optimize import minimize
import random

class BlackBoxOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.search_space = (-5.0, 5.0)
        self.func = lambda x: x[0] * x[1]  # Example black box function

    def __call__(self, func, initial_guess, iterations):
        def evaluate_fitness(individual):
            return func(individual)

        def mutate(individual):
            return [x + random.uniform(-0.01, 0.01) for x in individual]

        def __next__(self, iterations):
            fitness_values = [evaluate_fitness(individual) for individual in self.evaluate_individuals()]
            best_individual = self.best_individual
            best_fitness = min(fitness_values)
            for _ in range(min(iterations, self.budget)):
                for _ in range(self.dim):
                    new_individual = mutate(best_individual)
                    fitness_values = [evaluate_fitness(individual) for individual in self.evaluate_individuals()]
                    new_fitness = min(fitness_values)
                    if new_fitness > best_fitness:
                        best_individual = new_individual
                        best_fitness = new_fitness
                yield best_individual
            return best_individual

        def __next_multiple(self, iterations):
            fitness_values = [evaluate_fitness(individual) for individual in self.evaluate_individuals()]
            best_individual = self.best_individual
            best_fitness = min(fitness_values)
            for _ in range(iterations):
                for _ in range(self.dim):
                    new_individual = mutate(best_individual)
                    fitness_values = [evaluate_fitness(individual) for individual in self.evaluate_individuals()]
                    new_fitness = min(fitness_values)
                    if new_fitness > best_fitness:
                        best_individual = new_individual
                        best_fitness = new_fitness
                yield best_individual
            return best_individual

        self.best_individual = initial_guess
        self.budget -= 1

        return __next__(self.budget)

# One-line description with the main idea
# Novel metaheuristic algorithm for black box optimization using a novel search strategy