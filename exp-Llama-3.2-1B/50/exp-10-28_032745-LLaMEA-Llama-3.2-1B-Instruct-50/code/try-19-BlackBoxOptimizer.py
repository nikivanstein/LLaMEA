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
        self.population_size = 100
        self.population_mutations = 0.1

    def __call__(self, func, initial_guess, iterations):
        for _ in range(iterations):
            new_population = self.evaluate_new_population(func, initial_guess, self.budget)
            self.population = copy.deepcopy(new_population)
            for _ in range(self.population_size):
                if random.random() < 0.05:  # 5% mutation rate
                    self.population_mutations += 1
                    new_individual = copy.deepcopy(initial_guess)
                    new_individual[0] += random.uniform(-0.01, 0.01)
                    new_individual[1] += random.uniform(-0.01, 0.01)
                    self.population_mutations -= 1
                    self.population.append(new_individual)
            self.evaluate_fitness(self.population)
        return self.population

    def evaluate_new_population(self, func, initial_guess, budget):
        new_population = []
        for _ in range(budget):
            new_individual = copy.deepcopy(initial_guess)
            for i in range(self.dim):
                new_individual[i] += random.uniform(-0.01, 0.01)
            new_individual[0] *= new_individual[0]
            new_individual[1] *= new_individual[1]
            new_individual[0] *= new_individual[0]
            new_individual[1] *= new_individual[1]
            new_individual[0] = self.func(new_individual)
            new_individual[1] = self.func(new_individual)
            new_population.append(new_individual)
        return new_population

    def evaluate_fitness(self, population):
        best_individual = min(population, key=self.func)
        best_value = self.func(best_individual)
        print(f"Best individual: {best_individual}, Best value: {best_value}")

# One-line description with the main idea
# Novel metaheuristic algorithm for black box optimization using a novel search strategy