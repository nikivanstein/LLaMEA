import numpy as np
import random

class NovelizedStochasticMemeticAlgorithm:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 20
        self.mutation_rate = 0.1
        self.differential_evolution_params = {
            'CR': 0.5,
            'F': 0.5
        }
        self.population = self.initialize_population()

    def initialize_population(self):
        population = []
        for _ in range(self.population_size):
            individual = np.random.uniform(-5.0, 5.0, self.dim)
            population.append(individual)
        return population

    def evaluate(self, func):
        evaluations = []
        for individual in self.population:
            evaluation = func(individual)
            evaluations.append(evaluation)
        return evaluations

    def select_parents(self, evaluations):
        parents = []
        for _ in range(self.population_size):
            indices = np.argsort(evaluations)
            parents.append(self.population[indices[0]])
        return parents

    def crossover(self, parent1, parent2):
        child = parent1 + (parent2 - parent1) * random.uniform(0, 1)
        return child

    def mutate(self, individual):
        mutated_individual = individual.copy()
        for i in range(self.dim):
            if random.random() < self.mutation_rate:
                mutated_individual[i] += np.random.uniform(-1.0, 1.0)
        return mutated_individual

    def differential_evolution(self, parent1, parent2):
        child = parent1 + self.differential_evolution_params['F'] * (parent2 - parent1)
        return child

    def novelized_stochastic_memetic_algorithm(self, func):
        for _ in range(self.budget):
            evaluations = self.evaluate(func)
            parents = self.select_parents(evaluations)
            children = []
            for _ in range(self.population_size):
                parent1, parent2 = random.sample(parents, 2)
                child = self.crossover(parent1, parent2)
                child = self.mutate(child)
                children.append(child)
            self.population = children
            evaluations = self.evaluate(func)
            parents = self.select_parents(evaluations)
            for i in range(self.population_size):
                parent1, parent2 = random.sample(parents, 2)
                child = self.differential_evolution(parent1, parent2)
                self.population[i] = child
        best_individual = min(self.population, key=lambda individual: func(individual))
        return best_individual

# Example usage:
def func(x):
    return np.sum(x**2)

novelized_sma = NovelizedStochasticMemeticAlgorithm(budget=100, dim=10)
best_individual = novelized_sma(novelized_sma.__call__, func)
print(best_individual)