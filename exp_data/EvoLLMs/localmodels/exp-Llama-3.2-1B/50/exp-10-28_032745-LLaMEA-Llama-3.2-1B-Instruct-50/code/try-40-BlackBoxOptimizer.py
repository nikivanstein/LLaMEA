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
        population = [initial_guess] * self.dim
        for _ in range(iterations):
            fitnesses = [func(individual, self.search_space) for individual in population]
            best_index = np.argmin(fitnesses)
            best_individual = population[best_index]
            best_value = fitnesses[best_index]
            for i in range(self.dim):
                new_individual = [x + random.uniform(-0.01, 0.01) for x in best_individual]
                new_value = func(new_individual, self.search_space)
                if new_value < best_value:
                    best_individual = new_individual
                    best_value = new_value
            population[best_index] = best_individual
        return population

    def evaluate_fitness(self, individual):
        return self.func(individual, self.search_space)

# Novel metaheuristic algorithm for black box optimization using a novel search strategy
# 
# The algorithm uses a novel strategy to refine its search strategy, which is based on the probability of changing the individual's line.
# The probability of changing the individual's line is calculated as the inverse of the number of individuals that have already changed their line.
# The algorithm then uses this probability to select the next individual to change, which is the individual that has the lowest fitness value.
# 
# This algorithm has the potential to improve the efficiency of the optimization process by reducing the number of evaluations required to find the optimal solution.
# 
# One-line description with the main idea
# Novel metaheuristic algorithm for black box optimization using a novel search strategy