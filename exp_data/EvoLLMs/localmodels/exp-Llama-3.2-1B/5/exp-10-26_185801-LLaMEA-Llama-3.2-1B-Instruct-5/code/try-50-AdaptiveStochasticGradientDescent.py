import numpy as np
import random
import math

class AdaptiveStochasticGradientDescent:
    def __init__(self, budget, dim, learning_rate=0.01, tolerance=1e-6, max_iterations=1000):
        self.budget = budget
        self.dim = dim
        self.learning_rate = learning_rate
        self.tolerance = tolerance
        self.max_iterations = max_iterations
        self.search_space = np.linspace(-5.0, 5.0, dim)
        self.func_evaluations = 0
        self.new_individual = None

    def __call__(self, func):
        while self.func_evaluations < self.budget:
            self.func_evaluations += 1
            func_value = func(self.search_space)
            if np.isnan(func_value) or np.isinf(func_value):
                raise ValueError("Invalid function value")
            if func_value < 0 or func_value > 1:
                raise ValueError("Function value must be between 0 and 1")
            if self.func_evaluations / self.budget > self.tolerance:
                break
            self.search_space = np.linspace(-5.0, 5.0, self.dim)
        return func_value

    def mutate(self, individual):
        if random.random() < 0.5:
            self.new_individual = individual
            return True
        else:
            return False

    def evaluate_fitness(self, individual):
        func_value = self.__call__(individual)
        updated_individual = individual
        while self.func_evaluations < self.budget and updated_individual is None:
            updated_individual = self.f(updated_individual, self.logger)
        return updated_individual

    def f(self, individual, logger):
        func_value = self.__call__(individual)
        if self.mutate(individual):
            updated_individual = self.evaluate_fitness(individual)
        else:
            updated_individual = individual
        if self.func_evaluations / self.budget > self.tolerance:
            return updated_individual
        else:
            return func_value

    def initialize_single(self):
        self.new_individual = np.random.uniform(self.search_space)
        return self.new_individual

    def initialize_population(self, population_size):
        return [self.initialize_single() for _ in range(population_size)]

    def update_population(self, population, iterations):
        for _ in range(iterations):
            new_population = []
            for individual in population:
                new_individual = self.evaluate_fitness(individual)
                if new_individual is not None:
                    new_population.append(new_individual)
            population = new_population
        return population

# Description: Stochastic Gradient Descent with Adaptive Line Search for Black Box Optimization.
# Code: 
# ```python
# AdaptiveStochasticGradientDescent
# ```