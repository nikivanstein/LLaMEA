import numpy as np
import random

class AdaptiveBBOOPEvolutionaryStrategy:
    def __init__(self, budget, dim, mutation_rate, population_size, alpha, beta):
        self.budget = budget
        self.dim = dim
        self.search_space = np.linspace(-5.0, 5.0, dim)
        self.func_evaluations = 0
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.alpha = alpha
        self.beta = beta
        self.population = self.initialize_population()

    def initialize_population(self):
        return [np.random.uniform(self.search_space) for _ in range(self.population_size)]

    def __call__(self, func):
        while self.func_evaluations < self.budget:
            func_evaluations += 1
            if random.random() < self.mutation_rate:
                new_individual = self.population[random.randint(0, self.population_size - 1)]
                new_individual = self.evaluate_fitness(new_individual, func)
                if new_individual < 0 or new_individual > 1:
                    new_individual = self.adjust_individual(new_individual)
            new_individual = self.evaluate_fitness(new_individual, func)
            if new_individual < 0 or new_individual > 1:
                new_individual = self.adjust_individual(new_individual)
            self.population.append(new_individual)
            self.population = self.sort_population()
        return self.population[-1]

    def evaluate_fitness(self, individual, func):
        func_value = func(individual)
        if np.isnan(func_value) or np.isinf(func_value):
            raise ValueError("Invalid function value")
        if func_value < 0 or func_value > 1:
            raise ValueError("Function value must be between 0 and 1")
        return func_value

    def adjust_individual(self, individual):
        if individual < 0:
            individual = 0
        elif individual > 1:
            individual = 1
        return individual

    def sort_population(self):
        return sorted(self.population, key=self.evaluate_fitness)

    def mutate(self, individual):
        if random.random() < self.mutation_rate:
            return self.adjust_individual(individual)
        return individual

# Description: Adaptive Black Box Optimization using Evolutionary Strategies
# Code: 