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

class HSBBO:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.search_space = np.linspace(-5.0, 5.0, dim)
        self.func_evaluations = 0
        self.population = None
        self.fitness = None

    def __call__(self, func):
        while self.func_evaluations < self.budget:
            self.population = self.generate_population(func)
            fitness_values = self.evaluate_fitness(self.population, func)
            self.fitness = np.mean(fitness_values)
            if np.isnan(self.fitness) or np.isinf(self.fitness):
                raise ValueError("Invalid fitness value")
            if self.fitness < 0 or self.fitness > 1:
                raise ValueError("Fitness value must be between 0 and 1")
            self.search_space = np.linspace(-5.0, 5.0, self.dim)
        return func_value

    def generate_population(self, func):
        population = [func(random.uniform(self.search_space[0], self.search_space[1])) for _ in range(100)]
        return population

    def evaluate_fitness(self, population, func):
        fitness_values = []
        for individual in population:
            func_value = func(individual)
            fitness_values.append(func_value)
        return np.mean(fitness_values)

    def mutate(self, individual, mutation_rate):
        if random.random() < mutation_rate:
            index = random.randint(0, self.dim - 1)
            individual[index] = random.uniform(self.search_space[0], self.search_space[1])
        return individual

    def crossover(self, parent1, parent2):
        child = [parent1[i] + parent2[i] for i in range(self.dim)]
        return child

    def selection(self, population):
        fitness_values = np.array([self.evaluate_fitness(individual, func) for individual, func in zip(population, func)])
        return np.argsort(fitness_values)[::-1]

class HSBBO2:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.search_space = np.linspace(-5.0, 5.0, dim)
        self.func_evaluations = 0
        self.population = None
        self.fitness = None

    def __call__(self, func):
        while self.func_evaluations < self.budget:
            self.population = self.generate_population(func)
            fitness_values = self.evaluate_fitness(self.population, func)
            self.fitness = np.mean(fitness_values)
            if np.isnan(self.fitness) or np.isinf(self.fitness):
                raise ValueError("Invalid fitness value")
            if self.fitness < 0 or self.fitness > 1:
                raise ValueError("Fitness value must be between 0 and 1")
            self.search_space = np.linspace(-5.0, 5.0, self.dim)
        return func_value

    def generate_population(self, func):
        population = [func(random.uniform(self.search_space[0], self.search_space[1])) for _ in range(100)]
        return population

    def evaluate_fitness(self, population, func):
        fitness_values = []
        for individual in population:
            func_value = func(individual)
            fitness_values.append(func_value)
        return np.mean(fitness_values)

    def mutate(self, individual, mutation_rate):
        if random.random() < mutation_rate:
            index = random.randint(0, self.dim - 1)
            individual[index] = random.uniform(self.search_space[0], self.search_space[1])
        return individual

    def crossover(self, parent1, parent2):
        child = [parent1[i] + parent2[i] for i in range(self.dim)]
        return child

    def selection(self, population):
        fitness_values = np.array([self.evaluate_fitness(individual, func) for individual, func in zip(population, func)])
        return np.argsort(fitness_values)[::-1]

# Description: Novel Metaheuristic Algorithm for Black Box Optimization using Evolutionary Strategies
# Code: 