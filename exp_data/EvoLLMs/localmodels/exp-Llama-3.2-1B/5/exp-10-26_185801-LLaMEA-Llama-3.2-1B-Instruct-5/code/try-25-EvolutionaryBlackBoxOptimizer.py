import numpy as np
import random

class EvolutionaryBlackBoxOptimizer:
    def __init__(self, budget, dim, mutation_rate=0.01, elitism_rate=0.05):
        self.budget = budget
        self.dim = dim
        self.search_space = np.linspace(-5.0, 5.0, dim)
        self.func_evaluations = 0
        self.population_size = 100
        self.elite_size = 10
        self.mutation_rate = mutation_rate
        self.elitism_rate = elitism_rate
        self.population = self.initialize_population()

    def initialize_population(self):
        return [self.generate_individual() for _ in range(self.population_size)]

    def generate_individual(self):
        return np.random.uniform(self.search_space)

    def __call__(self, func):
        while self.func_evaluations < self.budget:
            func_evaluations += 1
            new_individual = self.evaluate_fitness(func, self.population, self.search_space)
            if np.isnan(new_individual) or np.isinf(new_individual):
                raise ValueError("Invalid function value")
            if new_individual < 0 or new_individual > 1:
                raise ValueError("Function value must be between 0 and 1")
            self.search_space = np.linspace(-5.0, 5.0, self.dim)
            if random.random() < self.elitism_rate:
                self.population[self.elite_size] = new_individual
            else:
                self.population.append(new_individual)
        return func(self.population[-1])

    def evaluate_fitness(self, func, population, search_space):
        return np.mean([func(x) for x in population])

# Example usage:
optimizer = EvolutionaryBlackBoxOptimizer(budget=100, dim=5)
print(optimizer(__call__(lambda x: x**2)))