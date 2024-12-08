import numpy as np
from scipy.optimize import minimize
import random
import math

class BlackBoxOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.search_space = (-5.0, 5.0)
        self.func = lambda x: x[0] * x[1]  # Example black box function

    def __call__(self, func, initial_guess, iterations):
        def evaluate_fitness(individual):
            return self.func(individual)

        def __evaluate_fitness(individual, budget):
            if budget <= 0:
                raise ValueError("Invalid number of function evaluations")
            if len(individual)!= self.dim:
                raise ValueError("Invalid number of individuals")

            fitness = evaluate_fitness(individual)
            for i in range(self.dim):
                new_individual = individual.copy()
                for _ in range(10):  # Adaptive strategy
                    new_individual[i] += random.uniform(-0.01, 0.01)
                    new_individual[i] = max(-5.0, min(5.0, new_individual[i]))
                    new_fitness = evaluate_fitness(new_individual)
                    if new_fitness < fitness:
                        fitness = new_fitness
                        individual = new_individual
            return fitness

        best_individual, best_fitness = None, None
        for _ in range(iterations):
            fitness = evaluate_fitness(initial_guess)
            if best_individual is None or best_fitness < fitness:
                best_individual = initial_guess
                best_fitness = fitness
            if _ >= self.budget:
                break
        return best_individual, best_fitness

# Novel Metaheuristic Algorithm for Black Box Optimization
# 
# One-line description with the main idea
# Novel metaheuristic algorithm for black box optimization using adaptive search strategy