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
        # Define the mutation strategy
        def mutate(individual):
            new_individual = individual.copy()
            for i in range(self.dim):
                if random.random() < 0.45:  # 45% chance of mutation
                    new_individual[i] += random.uniform(-0.01, 0.01)
            return new_individual

        # Evaluate the fitness of the current individual
        fitness = self.evaluate_fitness(individual)

        # Perform iterations
        for _ in range(iterations):
            if _ >= self.budget:
                break
            best_individual = initial_guess
            best_fitness = fitness
            for i in range(self.dim):
                new_individual = mutate(best_individual)
                new_fitness = self.evaluate_fitness(new_individual)
                if new_fitness < best_fitness:
                    best_individual = new_individual
                    best_fitness = new_fitness
            initial_guess = best_individual

        return best_individual, best_fitness

    def evaluate_fitness(self, individual):
        # Evaluate the fitness of the individual using the black box function
        # Replace this with your own implementation
        return self.func(individual)

# One-line description with the main idea
# Novel metaheuristic algorithm for black box optimization using a novel search strategy
# 
# Novel Metaheuristic Algorithm for Black Box Optimization
# ```python