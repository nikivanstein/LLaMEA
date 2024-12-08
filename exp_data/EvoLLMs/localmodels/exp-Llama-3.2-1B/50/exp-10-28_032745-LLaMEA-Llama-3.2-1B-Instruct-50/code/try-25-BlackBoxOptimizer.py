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
        
        def mutate(individual):
            return [x + random.uniform(-0.01, 0.01) for x in individual]
        
        def crossover(parent1, parent2):
            return [x + random.uniform(-0.01, 0.01) for x in parent1]
        
        def selection(population):
            return sorted(population, key=evaluate_fitness, reverse=True)[:int(self.budget/2)]
        
        population = [initial_guess]
        for _ in range(iterations):
            if _ >= self.budget:
                break
            parent1, parent2 = random.sample(population, 2)
            child1 = crossover(parent1, parent2)
            child2 = crossover(parent1, parent2)
            population.append(mutate(child1))
            population.append(mutate(child2))
        
        return population[0], evaluate_fitness(population[0])

# Novel metaheuristic algorithm for black box optimization using a novel search strategy
# 