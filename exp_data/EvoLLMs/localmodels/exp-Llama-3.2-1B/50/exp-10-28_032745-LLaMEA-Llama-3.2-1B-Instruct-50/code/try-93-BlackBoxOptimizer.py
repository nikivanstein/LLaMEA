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
            return self.func(individual)

        def fitness(individual):
            return evaluate_fitness(individual)

        best_x = initial_guess
        best_value = fitness(best_x)
        for _ in range(iterations):
            if _ >= self.budget:
                break
            for i in range(self.dim):
                new_x = [x + random.uniform(-0.01, 0.01) for x in best_x]
                new_value = evaluate_fitness(new_x)
                if new_value < best_value:
                    best_x = new_x
                    best_value = new_value
        return best_x, best_value

# Novel metaheuristic algorithm for black box optimization using a novel search strategy
# 
# 1. Initialize the population with random initial guesses
# 2. Evaluate the fitness of each individual in the population
# 3. Select the fittest individuals based on their fitness values
# 4. Refine the fittest individuals by applying a novel search strategy
# 5. Repeat steps 2-4 until a termination condition is met
# 
# One-line description with the main idea
# Novel metaheuristic algorithm for black box optimization using a novel search strategy
# 
# 
# ```python
# black_box_optimizer = BlackBoxOptimizer(1000, 10)
# best_individual, best_fitness = black_box_optimizer(BlackBoxOptimizer(1000, 10))
# print("Best individual:", best_individual)
# print("Best fitness:", best_fitness)
# ```