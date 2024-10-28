import random
import numpy as np

class AdaptiveBBOO:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.funcs = self.generate_functions()
        self.population = self.initialize_population()

    def generate_functions(self):
        functions = []
        for _ in range(24):
            func = lambda x: random.uniform(-5.0, 5.0)
            functions.append(func)
        return functions

    def initialize_population(self):
        population = []
        for _ in range(100):
            individual = np.random.uniform(-5.0, 5.0, self.dim)
            population.append(individual)
        return population

    def __call__(self, func, bounds, mutation_rate, crossover_rate):
        population = self.population
        while len(population) > 0:
            for _ in range(self.budget):
                new_individual = func(random.uniform(-5.0, 5.0), bounds)
                if random.random() < 0.4:
                    new_individual = random.uniform(bounds[0], bounds[1])
                if random.random() < 0.2:
                    new_individual = random.uniform(bounds[0], bounds[1])
                if random.random() < 0.4:
                    new_individual = random.uniform(bounds[0], bounds[1])
                new_individual = self.crossover(new_individual, population)
                new_individual = self.mutate(new_individual, bounds, mutation_rate, crossover_rate)
                population.append(new_individual)
            population = self.selection(population)
        return population

    def crossover(self, individual1, individual2):
        if random.random() < 0.5:
            return individual1[:len(individual2)//2] + individual2[len(individual2)//2:]
        else:
            return individual1 + individual2

    def mutate(self, individual, bounds, mutation_rate, crossover_rate):
        if random.random() < mutation_rate:
            if random.random() < crossover_rate:
                i = random.randint(0, len(individual) - 1)
                j = random.randint(0, len(individual) - 1)
                individual[i], individual[j] = individual[j], individual[i]
            return individual
        else:
            return individual

    def selection(self, population):
        fitnesses = [self.evaluate_fitness(individual) for individual in population]
        return np.argsort(fitnesses)[::-1][:self.budget]

    def evaluate_fitness(self, func, bounds, individual):
        return func(individual, bounds)

# Description: Adaptive Black Box Optimization using Genetic Algorithm with Mutation and Crossover
# Code: 