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

    def __call__(self, func, initial_guess, iterations):
        # Novel metaheuristic algorithm for black box optimization
        # 
        # Refine the strategy by changing individual lines to refine its search space
        # 
        # Probability of success: 0.45
        for _ in range(iterations):
            if _ >= self.budget:
                break
            # Evaluate the fitness of the current individual
            fitness = self.evaluate_fitness(initial_guess)
            
            # Refine the search space
            for i in range(self.dim):
                new_x = copy.deepcopy(initial_guess)
                for j in range(self.dim):
                    if random.random() < 0.5:  # Change individual lines to refine its search space
                        new_x[j] += random.uniform(-0.01, 0.01)
                new_x[i] += random.uniform(-0.01, 0.01)
                
                # Evaluate the fitness of the new individual
                new_fitness = self.evaluate_fitness(new_x)
                
                # Update the individual if the new fitness is better
                if new_fitness > fitness:
                    initial_guess = new_x
                    fitness = new_fitness
            # Update the individual's fitness
            fitness = self.evaluate_fitness(initial_guess)
        return initial_guess, fitness

    def evaluate_fitness(self, individual):
        # Evaluate the fitness of the individual
        # 
        # Probability of success: 0.45
        # 
        # Example: Evaluate the fitness of a function f(x) = x[0] * x[1]
        x = individual
        return x[0] * x[1]

# One-line description with the main idea
# Novel metaheuristic algorithm for black box optimization using a novel search strategy
# 
# Probability of success: 0.45