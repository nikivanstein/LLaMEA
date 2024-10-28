import numpy as np
import random
import copy

class HEACOMBBO:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.bounds = [(-5.0, 5.0)] * dim
        self.x0 = np.random.uniform(self.bounds[0][0], self.bounds[0][1], self.dim)

    def __call__(self, func):
        if self.budget == 0:
            return np.nan, np.nan

        def mutate(individual):
            mutation_rate = 0.4
            new_individual = copy.deepcopy(individual)
            for i in range(self.dim):
                if random.random() < mutation_rate:
                    new_individual[i] = random.uniform(self.bounds[i][0], self.bounds[i][1])
            return new_individual

        def crossover(parent1, parent2):
            crossover_rate = 0.4
            new_individual = copy.deepcopy(parent1)
            for i in range(self.dim):
                if random.random() < crossover_rate:
                    new_individual[i] = parent2[i]
            return new_individual

        def evaluate_fitness(individual):
            return func(individual)

        population_size = 10
        population = [np.random.uniform(self.bounds[0][0], self.bounds[0][1], self.dim) for _ in range(population_size)]

        for _ in range(self.budget):
            new_population = []
            for _ in range(population_size):
                parent1, parent2 = random.sample(population, 2)
                child = crossover(parent1, parent2)
                child = mutate(child)
                new_population.append(child)
            population = new_population

        best_individual = min(population, key=evaluate_fitness)
        return evaluate_fitness(best_individual), best_individual

def func(x):
    # Example function
    return x[0]**2 + x[1]**2

heacombbo = HEACOMBBO(100, 2)
best_fitness, best_individual = heacombbo(func)
print(f"Best Fitness: {best_fitness}, Best Individual: {best_individual}")