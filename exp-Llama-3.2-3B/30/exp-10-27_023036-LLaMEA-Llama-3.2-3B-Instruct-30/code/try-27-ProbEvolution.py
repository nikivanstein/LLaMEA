import numpy as np
import random

class ProbEvolution:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 100
        self.mutation_rate = 0.1
        self.crossover_rate = 0.7
        self.population = self.initialize_population()

    def initialize_population(self):
        population = []
        for _ in range(self.population_size):
            individual = np.random.uniform(-5.0, 5.0, self.dim)
            population.append(individual)
        return population

    def evaluate(self, func):
        for individual in self.population:
            func(individual)
        self.population = self.select_population()

    def select_population(self):
        # Select parents using tournament selection
        parents = []
        for _ in range(self.population_size):
            tournament = random.sample(self.population, 3)
            winner = max(tournament, key=lambda x: func(x))
            parents.append(winner)
        return parents

    def mutate(self, individual):
        if random.random() < self.mutation_rate:
            index = random.randint(0, self.dim - 1)
            individual[index] += np.random.uniform(-1.0, 1.0)
            if individual[index] < -5.0:
                individual[index] = -5.0
            elif individual[index] > 5.0:
                individual[index] = 5.0
        return individual

    def crossover(self, parent1, parent2):
        if random.random() < self.crossover_rate:
            index = random.randint(0, self.dim - 1)
            child = parent1.copy()
            child[index] = parent2[index]
            return child
        return parent1

    def optimize(self, func):
        for _ in range(self.budget):
            self.evaluate(func)
            new_population = []
            for _ in range(self.population_size):
                parent1 = random.choice(self.population)
                parent2 = random.choice(self.population)
                child = self.crossover(parent1, parent2)
                child = self.mutate(child)
                new_population.append(child)
            self.population = new_population

# Example usage:
def func(x):
    return np.sum(x**2)

prob_evolution = ProbEvolution(budget=100, dim=10)
prob_evolution.optimize(func)