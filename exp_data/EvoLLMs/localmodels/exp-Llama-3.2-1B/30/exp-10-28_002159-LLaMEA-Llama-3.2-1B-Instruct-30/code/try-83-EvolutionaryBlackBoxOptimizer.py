import random
import numpy as np

class EvolutionaryBlackBoxOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 50
        self.population = self.init_population()
        self.fitness_scores = np.zeros((self.population_size, self.dim))
        self.search_spaces = [(-5.0, 5.0)] * self.dim

    def init_population(self):
        population = []
        for _ in range(self.population_size):
            individual = [random.uniform(self.search_spaces[i][0], self.search_spaces[i][1]) for i in range(self.dim)]
            population.append(individual)
        return population

    def __call__(self, func):
        def fitness(individual):
            return func(individual)

        for _ in range(self.budget):
            for i, individual in enumerate(self.population):
                fitness_scores[i] = fitness(individual)
            best_individual = self.population[np.argmax(fitness_scores)]
            new_individual = self.evaluate_fitness(best_individual)
            if random.random() < 0.3:
                new_individual = self.mutate(new_individual)
            new_individual = np.array(new_individual)
            if fitness(individual) > fitness(new_individual):
                self.population[i] = new_individual
                self.fitness_scores[i] = fitness(new_individual)

        return self.population

    def mutate(self, individual):
        if random.random() < 0.01:
            index = random.randint(0, self.dim - 1)
            self.search_spaces[index] = (self.search_spaces[index][0] + random.uniform(-1, 1), self.search_spaces[index][1] + random.uniform(-1, 1))
        return individual

    def evaluate(self, func):
        return func(np.array(self.population))

def generate_new_individual(population_size, dim, budget):
    new_individual = np.zeros(dim)
    for i in range(dim):
        new_individual[i] = random.uniform(self.search_spaces[i][0], self.search_spaces[i][1])
    return new_individual

def recombine(parent1, parent2):
    new_individual = np.zeros((len(parent1), len(parent2)))
    for i in range(len(parent1)):
        for j in range(len(parent2)):
            new_individual[i, j] = (parent1[i, j] + parent2[i, j]) / 2
    return new_individual

def evaluateBBOB(func, population, budget):
    best_individual = population[np.argmax(func(population))]
    new_individual = generate_new_individual(population_size, dim, budget)
    if random.random() < 0.3:
        new_individual = recombine(best_individual, new_individual)
    return func(np.array([best_individual, new_individual]))

# Description: Evolutionary Black Box Optimization Algorithm
# Code: 