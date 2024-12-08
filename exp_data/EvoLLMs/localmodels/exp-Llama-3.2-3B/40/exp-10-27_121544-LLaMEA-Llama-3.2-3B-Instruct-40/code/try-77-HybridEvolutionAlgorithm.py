import numpy as np
import random
import operator

class HybridEvolutionAlgorithm:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.bounds = [(-5.0, 5.0)] * dim
        self.x0 = np.random.uniform(self.bounds[0][0], self.bounds[0][1], self.dim)
        self.population_size = 100
        self.mutation_rate = 0.2
        self.crossover_rate = 0.4
        self.logger = {}

    def __call__(self, func):
        if self.budget == 0:
            return np.nan, np.nan

        if func.__name__ == 'f1':
            self.population = self.initialize_population(func)
            for _ in range(self.budget):
                self.evolve_population(func)
            return self.fittest_individual(func)

        elif func.__name__ == 'f2':
            self.population = self.initialize_population(func)
            for _ in range(self.budget):
                self.evolve_population(func)
            return self.fittest_individual(func)

        elif func.__name__ == 'f3':
            self.population = self.initialize_population(func)
            for _ in range(self.budget):
                self.evolve_population(func)
            return self.fittest_individual(func)

        elif func.__name__ == 'f4':
            self.population = self.initialize_population(func)
            for _ in range(self.budget):
                self.evolve_population(func)
            return self.fittest_individual(func)

        elif func.__name__ == 'f5':
            self.population = self.initialize_population(func)
            for _ in range(self.budget):
                self.evolve_population(func)
            return self.fittest_individual(func)

        elif func.__name__ == 'f6':
            self.population = self.initialize_population(func)
            for _ in range(self.budget):
                self.evolve_population(func)
            return self.fittest_individual(func)

        elif func.__name__ == 'f7':
            self.population = self.initialize_population(func)
            for _ in range(self.budget):
                self.evolve_population(func)
            return self.fittest_individual(func)

        elif func.__name__ == 'f8':
            self.population = self.initialize_population(func)
            for _ in range(self.budget):
                self.evolve_population(func)
            return self.fittest_individual(func)

        elif func.__name__ == 'f9':
            self.population = self.initialize_population(func)
            for _ in range(self.budget):
                self.evolve_population(func)
            return self.fittest_individual(func)

        elif func.__name__ == 'f10':
            self.population = self.initialize_population(func)
            for _ in range(self.budget):
                self.evolve_population(func)
            return self.fittest_individual(func)

        elif func.__name__ == 'f11':
            self.population = self.initialize_population(func)
            for _ in range(self.budget):
                self.evolve_population(func)
            return self.fittest_individual(func)

        elif func.__name__ == 'f12':
            self.population = self.initialize_population(func)
            for _ in range(self.budget):
                self.evolve_population(func)
            return self.fittest_individual(func)

        elif func.__name__ == 'f13':
            self.population = self.initialize_population(func)
            for _ in range(self.budget):
                self.evolve_population(func)
            return self.fittest_individual(func)

        elif func.__name__ == 'f14':
            self.population = self.initialize_population(func)
            for _ in range(self.budget):
                self.evolve_population(func)
            return self.fittest_individual(func)

        elif func.__name__ == 'f15':
            self.population = self.initialize_population(func)
            for _ in range(self.budget):
                self.evolve_population(func)
            return self.fittest_individual(func)

        elif func.__name__ == 'f16':
            self.population = self.initialize_population(func)
            for _ in range(self.budget):
                self.evolve_population(func)
            return self.fittest_individual(func)

        elif func.__name__ == 'f17':
            self.population = self.initialize_population(func)
            for _ in range(self.budget):
                self.evolve_population(func)
            return self.fittest_individual(func)

        elif func.__name__ == 'f18':
            self.population = self.initialize_population(func)
            for _ in range(self.budget):
                self.evolve_population(func)
            return self.fittest_individual(func)

        elif func.__name__ == 'f19':
            self.population = self.initialize_population(func)
            for _ in range(self.budget):
                self.evolve_population(func)
            return self.fittest_individual(func)

        elif func.__name__ == 'f20':
            self.population = self.initialize_population(func)
            for _ in range(self.budget):
                self.evolve_population(func)
            return self.fittest_individual(func)

        elif func.__name__ == 'f21':
            self.population = self.initialize_population(func)
            for _ in range(self.budget):
                self.evolve_population(func)
            return self.fittest_individual(func)

        elif func.__name__ == 'f22':
            self.population = self.initialize_population(func)
            for _ in range(self.budget):
                self.evolve_population(func)
            return self.fittest_individual(func)

        elif func.__name__ == 'f23':
            self.population = self.initialize_population(func)
            for _ in range(self.budget):
                self.evolve_population(func)
            return self.fittest_individual(func)

        elif func.__name__ == 'f24':
            self.population = self.initialize_population(func)
            for _ in range(self.budget):
                self.evolve_population(func)
            return self.fittest_individual(func)

    def initialize_population(self, func):
        population = []
        for _ in range(self.population_size):
            individual = np.random.uniform(self.bounds[0][0], self.bounds[0][1], self.dim)
            population.append(individual)
        return population

    def evolve_population(self, func):
        new_population = []
        for individual in self.population:
            fitness = func(individual)
            if np.random.rand() < self.crossover_rate:
                parent1 = np.random.choice(self.population)
                parent2 = np.random.choice(self.population)
                child = self.crossover(parent1, parent2)
                if np.random.rand() < self.mutation_rate:
                    child = self.mutate(child)
            else:
                child = individual
            new_population.append(child)
        self.population = new_population

    def crossover(self, parent1, parent2):
        child = []
        for i in range(self.dim):
            if np.random.rand() < 0.5:
                child.append(parent1[i])
            else:
                child.append(parent2[i])
        return child

    def mutate(self, individual):
        mutated_individual = individual.copy()
        for i in range(self.dim):
            if np.random.rand() < self.mutation_rate:
                mutated_individual[i] += np.random.uniform(-1, 1)
        return mutated_individual

    def fittest_individual(self, func):
        fitness = []
        for individual in self.population:
            fitness.append(func(individual))
        fittest_index = np.argmin(fitness)
        fittest_individual = self.population[fittest_index]
        return fittest_individual, np.min(fitness)

# Usage:
# ```python
# bbo = HybridEvolutionAlgorithm(100, 10)
# best_individual, best_fitness = bbo('f1')
# print(best_individual, best_fitness)
# ```