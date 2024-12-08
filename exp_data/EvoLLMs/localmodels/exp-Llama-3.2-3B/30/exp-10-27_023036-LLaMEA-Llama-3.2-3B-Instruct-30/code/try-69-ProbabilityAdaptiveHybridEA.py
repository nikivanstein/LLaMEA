import numpy as np
import random

class ProbabilityAdaptiveHybridEA:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 100
        self.pso = PSO(self.population_size, self.dim, 0.8, 0.4)
        self.de = DE(self.population_size, self.dim, 0.5, 0.2)
        self.ga = GA(self.population_size, self.dim, 0.5, 0.1)
        self.population = self.initialize_population()
        self.best_individual = None

    def __call__(self, func):
        for _ in range(self.budget):
            # Selection
            population = self.population
            weights = np.random.uniform(0, 1, size=self.population_size)
            weights /= weights.sum()
            selected = np.random.choice(self.population_size, size=self.population_size, p=weights)
            selected = [population[i] for i in selected]

            # Crossover
            offspring = []
            for _ in range(self.population_size // 2):
                parent1, parent2 = random.sample(selected, 2)
                child = self.de.crossover(parent1, parent2)
                offspring.append(child)

            # Mutation
            for i in range(len(offspring)):
                if random.random() < 0.3:
                    offspring[i] = self.pso.mutation(offspring[i])

            # Hybridization
            for i in range(len(offspring)):
                if random.random() < 0.3:
                    offspring[i] = self.ga.mutate(offspring[i])

            # Evaluation
            for i in range(len(offspring)):
                offspring[i] = func(offspring[i])

            # Replacement
            self.population = offspring
            self.population = [self.pso.select(self.population) for _ in range(self.population_size)]

            # Update best individual
            if self.best_individual is None or self.best_individual[1] < min(offspring):
                self.best_individual = (min(offspring), min(offspring))

    def initialize_population(self):
        population = []
        for _ in range(self.population_size):
            individual = np.random.uniform(-5.0, 5.0, self.dim)
            population.append(individual)
        return population


class PSO:
    def __init__(self, population_size, dim, c1, c2):
        self.population_size = population_size
        self.dim = dim
        self.c1 = c1
        self.c2 = c2
        self.v = np.zeros((self.population_size, self.dim))
        self.x = np.random.uniform(-5.0, 5.0, (self.population_size, self.dim))
        self.best = np.zeros((self.population_size, self.dim))

    def update(self, func):
        for i in range(self.population_size):
            r1 = np.random.uniform(0, 1)
            r2 = np.random.uniform(0, 1)
            self.x[i] += self.c1 * r1 * (self.best[i] - self.x[i]) + self.c2 * r2 * (self.x[i] - self.x[self.get_best()])
            self.v[i] += self.c1 * r1 * (self.best[i] - self.x[i]) + self.c2 * r2 * (self.x[i] - self.x[self.get_best()])
            self.x[i] += self.v[i]
            self.best[i] = self.x[i]
            self.best[i] = func(self.best[i])

    def get_best(self):
        return np.argmin([self.best[i, :].mean() for i in range(self.population_size)])

    def select(self, population):
        weights = np.random.uniform(0, 1, size=self.population_size)
        weights /= weights.sum()
        selected = np.random.choice(self.population_size, size=self.population_size, p=weights)
        return [population[i] for i in selected]

    def mutation(self, individual):
        for i in range(self.dim):
            if random.random() < 0.3:
                individual[i] += np.random.uniform(-0.5, 0.5)
        return individual


class DE:
    def __init__(self, population_size, dim, c1, c2):
        self.population_size = population_size
        self.dim = dim
        self.c1 = c1
        self.c2 = c2
        self.x = np.random.uniform(-5.0, 5.0, (self.population_size, self.dim))
        self.f = np.zeros((self.population_size, self.dim))

    def update(self, func):
        for i in range(self.population_size):
            r1 = np.random.uniform(0, 1)
            r2 = np.random.uniform(0, 1)
            r3 = np.random.uniform(0, 1)
            f1 = self.x[i] + r1 * (self.x[self.get_best(i)] - self.x[i])
            f2 = self.x[i] + r2 * (self.x[self.get_best(i)] - self.x[i]) + r3 * (self.f[self.get_best(i)] - self.f[i])
            self.x[i] = f1
            self.f[i] = func(self.x[i])
            if self.f[i] < self.f[self.get_best(i)]:
                self.x[i] = f2
                self.f[i] = func(self.x[i])

    def get_best(self, i):
        return np.argmin([self.f[j, :].mean() for j in range(self.population_size) if j!= i])

    def crossover(self, parent1, parent2):
        child = parent1 + 0.5 * (parent2 - parent1)
        return child

    def mutate(self, individual):
        for i in range(self.dim):
            if random.random() < 0.3:
                individual[i] += np.random.uniform(-0.5, 0.5)
        return individual


class GA:
    def __init__(self, population_size, dim, mutation_rate, crossover_rate):
        self.population_size = population_size
        self.dim = dim
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.x = np.random.uniform(-5.0, 5.0, (self.population_size, self.dim))
        self.f = np.zeros((self.population_size, self.dim))

    def update(self, func):
        for i in range(self.population_size):
            if random.random() < self.mutation_rate:
                self.x[i] = self.mutate(self.x[i])
            if random.random() < self.crossover_rate:
                self.x[i] = self.crossover(self.x[i], self.x[self.get_best(i)])
            self.f[i] = func(self.x[i])

    def get_best(self):
        return np.argmin([self.f[i, :].mean() for i in range(self.population_size)])

    def mutate(self, individual):
        for i in range(self.dim):
            if random.random() < self.mutation_rate:
                individual[i] += np.random.uniform(-0.5, 0.5)
        return individual

    def crossover(self, parent1, parent2):
        child = 0.5 * (parent1 + parent2)
        return child

# Usage
def func(x):
    return np.sum(x**2)

ea = ProbabilityAdaptiveHybridEA(100, 10)
ea()