import random
import numpy as np

class AdaptiveGeneticAlgorithm:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.funcs = self.generate_functions()
        self.population = self.initialize_population()
        self.fitness_scores = self.calculate_fitness_scores()

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

    def calculate_fitness_scores(self):
        fitness_scores = []
        for func in self.funcs:
            individual = self.population[0]
            for _ in range(self.budget):
                func(individual)
                if random.random() < 0.4:
                    individual = random.uniform(-5.0, 5.0)
            fitness_scores.append(np.mean([func(individual) for func in self.funcs]))
        return fitness_scores

    def __call__(self, func, x0, bounds, budget):
        individual = x0
        for _ in range(budget):
            func(individual)
            if random.random() < 0.5:
                individual = random.uniform(bounds[0], bounds[1])
            if random.random() < 0.2:
                individual = random.uniform(bounds[0], bounds[1])
            if random.random() < 0.4:
                individual = random.uniform(bounds[0], bounds[1])
        return func(individual)

# Example usage:
ga = AdaptiveGeneticAlgorithm(100, 10)
ga.population = ga.initialize_population()
ga.fitness_scores = ga.calculate_fitness_scores()
ga.__call__(ga.funcs[0], ga.population[0], [(-5.0, 5.0)], 100)