import random
import numpy as np

class AdaptiveEvolutionStrategy:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.funcs = self.generate_functions()
        self.population = self.generate_population()
        self.fitnesses = self.generate_fitnesses()
        self.algorithms = [BBOB(self.budget, self.dim) for _ in range(10)]

    def generate_functions(self):
        functions = []
        for _ in range(24):
            func = lambda x: random.uniform(-5.0, 5.0)
            functions.append(func)
        return functions

    def generate_population(self):
        population = []
        for _ in range(100):
            individual = [random.uniform(-5.0, 5.0) for _ in range(self.dim)]
            population.append(individual)
        return population

    def generate_fitnesses(self):
        fitnesses = []
        for individual in self.population:
            fitness = self.f(individual, self.funcs)
            fitnesses.append(fitness)
        return fitnesses

    def __call__(self, func):
        best_individual = None
        best_fitness = -np.inf
        for algorithm in self.algorithms:
            fitness = algorithm(func, random.uniform(-5.0, 5.0), [random.uniform(-5.0, 5.0) for _ in range(self.dim)], self.budget)
            if fitness > best_fitness:
                best_fitness = fitness
                best_individual = individual
        return best_individual, best_fitness

    def mutate(self, individual):
        mutated_individual = [x + random.uniform(-0.1, 0.1) for x in individual]
        return mutated_individual

    def evaluate_fitness(self, individual, bounds, budget):
        fitness = self.f(individual, bounds)
        if random.random() < 0.4:
            individual = self.mutate(individual)
        if random.random() < 0.2:
            bounds = self.mutate(bounds)
        if random.random() < 0.4:
            bounds = self.mutate(bounds)
        return fitness

    def bbo_opt(self, func, x0, bounds, budget):
        x = x0
        for _ in range(budget):
            x = func(x)
            if x < bounds[0]:
                x = bounds[0]
            elif x > bounds[1]:
                x = bounds[1]
            if random.random() < 0.5:
                x = random.uniform(bounds[0], bounds[1])
            if random.random() < 0.2:
                x = random.uniform(bounds[0], bounds[1])
            if random.random() < 0.4:
                x = random.uniform(bounds[0], bounds[1])
        return x

# Description: Adaptive Evolution Strategy using Black Box Optimization
# Code: 