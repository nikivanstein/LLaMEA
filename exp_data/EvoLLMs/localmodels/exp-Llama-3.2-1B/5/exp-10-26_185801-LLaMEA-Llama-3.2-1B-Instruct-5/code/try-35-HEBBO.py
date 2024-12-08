import numpy as np
import random

class HEBBO:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.search_space = np.linspace(-5.0, 5.0, dim)
        self.func_evaluations = 0

    def __call__(self, func):
        while self.func_evaluations < self.budget:
            func_evaluations += 1
            func_value = func(self.search_space)
            if np.isnan(func_value) or np.isinf(func_value):
                raise ValueError("Invalid function value")
            if func_value < 0 or func_value > 1:
                raise ValueError("Function value must be between 0 and 1")
            self.search_space = np.linspace(-5.0, 5.0, self.dim)
        return func_value

    def __str__(self):
        return f"HeBO: {self.budget} function evaluations"

class HEBBOMetaheuristic:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.search_space = np.linspace(-5.0, 5.0, dim)
        self.func_evaluations = 0

    def __call__(self, func, population_size, mutation_rate, num_generations):
        population = self.generate_population(population_size)
        for generation in range(num_generations):
            new_population = self.evaluate_fitness(population, func)
            self.update_population(new_population, population_size, mutation_rate)
            if generation % 10 == 0:
                print(f"Generation {generation}: {self.__str__()}")

    def generate_population(self, population_size):
        return [np.random.uniform(self.search_space) for _ in range(population_size)]

    def evaluate_fitness(self, population, func):
        fitness_values = []
        for individual in population:
            fitness_value = func(individual)
            fitness_values.append(fitness_value)
        return fitness_values

    def update_population(self, new_population, population_size, mutation_rate):
        for i in range(population_size):
            parent1, parent2 = random.sample(new_population, 2)
            child = (parent1 + parent2) / 2
            if random.random() < mutation_rate:
                child[0] += random.uniform(-1, 1)
                child[1] += random.uniform(-1, 1)
            new_population[i] = child

    def __str__(self):
        return f"HEBBOMetaheuristic: {self.budget} function evaluations"

# Description: Novel Metaheuristic Algorithm for Black Box Optimization
# Code: 