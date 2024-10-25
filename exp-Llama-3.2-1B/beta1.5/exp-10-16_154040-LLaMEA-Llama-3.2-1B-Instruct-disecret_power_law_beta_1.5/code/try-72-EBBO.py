# Description: Evolutionary Black Box Optimization using Genetic Algorithm
# Code: 
import random
import numpy as np

class EBBO:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 50
        self.population = self.initialize_population()

    def initialize_population(self):
        population = []
        for _ in range(self.population_size):
            dim = self.dim * random.random()
            func = self.generate_func(dim)
            population.append((func, random.uniform(-5.0, 5.0), random.uniform(-5.0, 5.0)))
        return population

    def generate_func(self, dim):
        return np.sin(np.sqrt(dim))

    def __call__(self, func):
        for _ in range(self.budget):
            fitness = self.evaluate(func)
            if fitness < 0:
                break
        return func, fitness

    def evaluate(self, func):
        return func(func, random.uniform(-5.0, 5.0), random.uniform(-5.0, 5.0))

    def __next__(self):
        while True:
            new_individual = self.evaluate(func=self.evaluate(func))
            fitness = new_individual[1]
            if fitness < 0:
                break
            new_individual = (new_individual[0], self.evaluate(func=new_individual[0]))
            population = self.population + [new_individual]
            population = self.select(population, self.budget)
            population = self.bubble(population, self.dim)
            population = self.crossover(population)
            population = self.mutate(population)
            population = self.evaluate(func=self.evaluate(func))
            if len(population) == 1:
                return population[0]
            return random.choice(population)

    def select(self, population, budget):
        # Select parents using tournament selection
        tournament_size = int(budget / 2)
        winners = []
        for _ in range(budget):
            individual1, individual2 = random.sample(population, 2)
            winner = max(individual1, key=lambda x: x[1])
            if winner[1] < individual2[1]:
                winners.append(winner)
            else:
                winners.append(individual2)
        return winners

    def bubble(self, population, dim):
        # Bubble sort
        for _ in range(len(population)):
            for i in range(len(population) - 1):
                if population[i][1] > population[i + 1][1]:
                    population[i], population[i + 1] = population[i + 1], population[i]
        return population

    def crossover(self, population):
        # Crossover
        children = []
        while len(population) > 1:
            parent1, parent2 = random.sample(population, 2)
            child1, child2 = self.evaluate(func=parent1), self.evaluate(func=parent2)
            if child1[1] < child2[1]:
                children.append(child1)
            else:
                children.append(child2)
        return children

    def mutate(self, population):
        # Mutate
        mutated_population = []
        for individual in population:
            mutated_individual = (individual[0], individual[1] + random.uniform(-1.0, 1.0))
            mutated_population.append(mutated_individual)
        return mutated_population

# One-line description: Evolutionary Black Box Optimization using Genetic Algorithm
# Code: 