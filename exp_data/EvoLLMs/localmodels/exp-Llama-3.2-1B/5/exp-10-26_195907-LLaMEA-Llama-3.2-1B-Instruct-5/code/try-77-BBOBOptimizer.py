import random
import numpy as np

class BBOBOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.search_space = 2 * np.random.uniform(-5.0, 5.0, size=(dim, 2))
        self.func = lambda x: np.sum(x)

    def __call__(self, func):
        population = [self.evaluate_fitness(individual, self.budget, func) for individual in self.generate_initial_population(self.dim)]
        while True:
            new_population = []
            for _ in range(self.budget):
                parent1, parent2 = random.sample(population, 2)
                child = self.crossover(parent1, parent2)
                child = self.mutation(child, func)
                new_population.append(child)
            population = new_population
            population = self.select_parents(population, self.budget)
        return self.evaluate_fitness(population[0], self.budget, func)

    def generate_initial_population(self, dim):
        return np.random.uniform(self.search_space, size=(dim, 2))

    def crossover(self, parent1, parent2):
        if random.random() < 0.5:
            return np.vstack((parent1[:dim, :], parent2[dim:, :]))
        else:
            return np.vstack((parent1, parent2[:dim, :]))

    def mutation(self, individual):
        if random.random() < 0.05:
            idx = np.random.randint(0, individual.shape[0])
            individual[idx] = random.uniform(-5.0, 5.0)
        return individual

    def select_parents(self, population, budget):
        # Select parents using tournament selection
        tournament_size = 2
        winners = []
        for _ in range(budget):
            individual1, individual2 = random.sample(population, tournament_size)
            winner = np.sum(individual1) / (winner1 + winner2)
            winners.append((individual1, winner))
        winners = np.array(winners)
        winners = winners[np.argsort(winners[:, 1])]
        return winners[:, 0]

# Genetic Algorithm
class GeneticAlgorithm:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.search_space = 2 * np.random.uniform(-5.0, 5.0, size=(dim, 2))
        self.func = lambda x: np.sum(x)

    def __call__(self, func):
        population = [self.evaluate_fitness(individual, self.budget, func) for individual in self.generate_initial_population(self.dim)]
        while True:
            new_population = []
            for _ in range(self.budget):
                parent1, parent2 = random.sample(population, 2)
                child = self.crossover(parent1, parent2)
                child = self.mutation(child, func)
                new_population.append(child)
            population = new_population
            population = self.select_parents(population, self.budget)
        return self.evaluate_fitness(population[0], self.budget, func)

# Description: Genetic Algorithm
# Code: 