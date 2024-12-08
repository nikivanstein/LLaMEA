import random
import numpy as np

class AdaptiveBBOB:
    def __init__(self, budget, dim, alpha, beta, gamma):
        self.budget = budget
        self.dim = dim
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.funcs = self.generate_functions()

    def generate_functions(self):
        functions = []
        for _ in range(24):
            func = lambda x: np.random.uniform(-5.0, 5.0, self.dim)
            functions.append(func)
        return functions

    def __call__(self, func, x0, bounds, budget):
        # Initialize population
        population = [x0] * self.budget
        for _ in range(self.budget):
            population[_] = func(population[_])

        # Calculate fitness
        fitnesses = [self.fitness(func, population) for func in self.funcs]

        # Select parents
        parents = []
        for _ in range(self.budget // 2):
            parent1, parent2 = random.sample(population, 2)
            fitnesses[parent1], fitnesses[parent2] = fitnesses[parent2], fitnesses[parent1]
            parents.append((parent1, fitnesses[parent1]))

        # Crossover
        children = []
        for _ in range(self.budget // 2):
            parent1, fitness1 = parents[_]
            parent2, fitness2 = parents[_ + 1]
            child1 = self.crossover(parent1, parent2, fitness1)
            child2 = self.crossover(parent1, parent2, fitness2)
            children.extend([child1, child2])

        # Mutate
        for i in range(self.budget):
            if random.random() < self.alpha:
                child = self.mutate(child, population[i])
                children[i] = child

        # Replace with new generation
        population = children

        # Evaluate fitness
        fitnesses = [self.fitness(func, population) for func in self.funcs]

        # Select fittest individuals
        fittest = sorted(zip(population, fitnesses), key=lambda x: x[1], reverse=True)[:self.budget // 2]

        # Replace with new generation
        population = [x[0] for x in fittest]

        # Update bounds
        bounds = [x[1][0] for x in fittest]

        return population, bounds, fitnesses

    def fitness(self, func, population):
        return np.mean([func(x) for x in population])

    def crossover(self, parent1, parent2, fitness):
        x1, x2 = parent1
        x2 = parent2
        if random.random() < self.beta:
            x1 = self.mutate(x1, fitness)
        if random.random() < self.gamma:
            x2 = self.mutate(x2, fitness)
        return x1, x2

    def mutate(self, x, fitness):
        if random.random() < self.alpha:
            x = self.better(x, fitness)
        return x

    def better(self, x, fitness):
        return x + np.random.uniform(-1, 1)

# One-line description: Evolutionary Algorithm for Multi-Dimensional Optimization using Adaptive Step Size Control
# Code: 