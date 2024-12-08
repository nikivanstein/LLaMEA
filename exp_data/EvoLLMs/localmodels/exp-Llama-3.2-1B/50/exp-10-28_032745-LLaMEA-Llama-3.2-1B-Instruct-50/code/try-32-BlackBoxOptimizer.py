import numpy as np
from scipy.optimize import minimize
import random
import copy

class BlackBoxOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.search_space = (-5.0, 5.0)
        self.func = lambda x: x[0] * x[1]  # Example black box function
        self.iterations = 100
        self.population_size = 100
        self.population = self.initialize_population()

    def initialize_population(self):
        return [copy.deepcopy(self.func(np.array([random.uniform(-5.0, 5.0), random.uniform(-5.0, 5.0)]))) for _ in range(self.population_size)]

    def __call__(self, func, initial_guess, iterations):
        for _ in range(iterations):
            if _ >= self.budget:
                break
            best_x = initial_guess
            best_value = self.func(best_x)
            for i in range(self.dim):
                new_x = [x + random.uniform(-0.01, 0.01) for x in best_x]
                new_value = self.func(new_x)
                if new_value < best_value:
                    best_x = new_x
                    best_value = new_value
            self.population = [copy.deepcopy(func(x)) for x in best_x]
        return self.evaluate_fitness(self.population)

    def evaluate_fitness(self, population):
        fitness = [self.func(x) for x in population]
        return fitness

    def mutate(self, individual):
        if random.random() < 0.45:
            return individual + random.uniform(-0.01, 0.01)
        return individual

# Novel Metaheuristic Algorithm for Black Box Optimization
# 
# Novel metaheuristic algorithm for black box optimization using a novel search strategy
# 
# Exception: Traceback (most recent call last):
#   File "/root/LLaMEA/llamea/llamea.py", line 187, in initialize_single
#     new_individual = self.evaluate_fitness(new_individual)
#   File "/root/LLaMEA/llamea/llamea.py", line 264, in evaluate_fitness
#     updated_individual = self.f(individual, self.logger)
#   File "/root/LLaMEA/llamea/llamea.py", line 264, in evaluate_fitness
#     updated_individual = self.f(individual, self.logger)
# 
# One-line description with the main idea
# Novel metaheuristic algorithm for black box optimization using a novel search strategy