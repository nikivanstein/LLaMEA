# Description: Evolutionary Multi-Optimization using Evolved Pareto Optimal (EPO) Algorithm
# Code: 
import random
import math
import operator
import copy

class EPO:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.funcs = self.generate_functions()
        self.population = self.initialize_population()
        self.fitnesses = self.initialize_fitnesses()

    def generate_functions(self):
        functions = []
        for _ in range(24):
            func = lambda x: random.uniform(-5.0, 5.0)
            functions.append(func)
        return functions

    def initialize_population(self):
        population = []
        for _ in range(100):
            individual = [random.uniform(-5.0, 5.0) for _ in range(self.dim)]
            population.append(individual)
        return population

    def initialize_fitnesses(self):
        fitnesses = []
        for individual in population:
            fitness = 0
            for func in self.funcs:
                fitness += func(individual)
            fitnesses.append(fitness)
        return fitnesses

    def __call__(self, func, bounds, budget):
        new_population = self.population[:]
        for _ in range(budget):
            for individual in new_population:
                fitness = self.fitnesses[individual]
                if random.random() < 0.4:
                    # Refine the individual using the EPO strategy
                    bounds = self.evolved_bounds(individual, bounds)
                    individual = self.evolve(individual, bounds, func)
                fitnesses[individual] = fitness
            new_population = self.select(population, fitnesses, bounds, budget)
        return new_population

    def evolved_bounds(self, individual, bounds):
        bounds = copy.deepcopy(bounds)
        for func in self.funcs:
            new_bounds = [func(individual[i]) for i in range(self.dim)]
            if random.random() < 0.4:
                new_bounds = [bounds[i] + random.uniform(-0.1, 0.1) for i in range(self.dim)]
            if random.random() < 0.2:
                new_bounds = [bounds[i] + random.uniform(-0.1, 0.1) for i in range(self.dim)]
            if random.random() < 0.4:
                new_bounds = [bounds[i] - random.uniform(-0.1, 0.1) for i in range(self.dim)]
            bounds = new_bounds
        return bounds

    def evolve(self, individual, bounds, func):
        x = individual
        for _ in range(100):
            x = func(x)
            if x < bounds[0]:
                x = bounds[0]
            elif x > bounds[1]:
                x = bounds[1]
            if random.random() < 0.2:
                x = random.uniform(bounds[0], bounds[1])
            if random.random() < 0.4:
                x = random.uniform(bounds[0], bounds[1])
            if random.random() < 0.6:
                x = random.uniform(bounds[0], bounds[1])
        return x

    def select(self, population, fitnesses, bounds, budget):
        new_population = []
        for _ in range(budget):
            fitnesses.sort(key=lambda x: x[-1], reverse=True)
            new_population.append(population[0])
            for individual in population[1:]:
                fitness = fitnesses[individual]
                if random.random() < 0.4:
                    # Select individuals based on their fitness
                    new_individual = copy.deepcopy(individual)
                    for func in self.funcs:
                        new_individual[func(new_individual)] = func(individual[func(new_individual)])
                    new_population.append(new_individual)
                else:
                    # Select individuals based on their fitness and bounds
                    bounds = copy.deepcopy(bounds)
                    for func in self.funcs:
                        new_bounds = [func(individual[i]) for i in range(self.dim)]
                        if random.random() < 0.4:
                            new_bounds = [bounds[i] + random.uniform(-0.1, 0.1) for i in range(self.dim)]
                        if random.random() < 0.2:
                            new_bounds = [bounds[i] + random.uniform(-0.1, 0.1) for i in range(self.dim)]
                        if random.random() < 0.4:
                            new_bounds = [bounds[i] - random.uniform(-0.1, 0.1) for i in range(self.dim)]
                        new_bounds = [new_bounds[i] for i in range(self.dim)]
                        new_individual = [new_bounds[i] for i in range(self.dim)]
                        new_individual[func(new_individual)] = func(individual[func(new_individual)])
                        new_population.append(new_individual)
                    new_population.sort(key=lambda x: x[-1], reverse=True)
                    new_population = new_population[:budget]
            new_population.append(new_population[0])
        return new_population

# Description: Evolutionary Multi-Optimization using Evolved Pareto Optimal (EPO) Algorithm
# Code: 