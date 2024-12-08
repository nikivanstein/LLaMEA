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
        population_size = 100
        population = [initial_guess] * population_size
        for _ in range(iterations):
            for i in range(population_size):
                fitness = self.func(population[i])
                if fitness < self.func(population[(i+1) % population_size]):
                    population[(i+1) % population_size], population[i] = population[i], population[(i+1) % population_size]
        return population[0], population[0]

    def select_strategy(self, initial_guess, iterations):
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
            initial_guess = best_x
        return initial_guess

    def mutate(self, individual, mutation_rate):
        if random.random() < mutation_rate:
            i, j = random.sample(range(len(individual)), 2)
            individual[i], individual[j] = individual[j], individual[i]
        return individual

# Novel metaheuristic algorithm for black box optimization using a novel search strategy
# 
# One-line description with the main idea
# Novel metaheuristic algorithm for black box optimization using a novel search strategy
# 