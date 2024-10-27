import random
import numpy as np

class TreeStructuredGeneticAlgorithm:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 10
        self.tree_size = 5
        self.mutation_rate = 0.35
        self.crossover_rate = 0.5
        self.population = self._initialize_population()

    def _initialize_population(self):
        population = []
        for _ in range(self.population_size):
            tree = {}
            for i in range(self.tree_size):
                tree[i] = {'lower': -5.0, 'upper': 5.0, 'value': random.uniform(-5.0, 5.0)}
            population.append(tree)
        return population

    def __call__(self, func):
        for _ in range(self.budget):
            self._evaluate_and_mutate_population(func)

    def _evaluate_and_mutate_population(self, func):
        fitnesses = []
        for individual in self.population:
            fitness = func(individual)
            fitnesses.append(fitness)
        min_fitness = min(fitnesses)
        min_index = fitnesses.index(min_fitness)
        self.population[min_index] = self._mutate_and_crossover(individuals=[self.population[min_index]])

    def _mutate_and_crossover(self, individual):
        mutated_individual = individual.copy()
        if random.random() < self.mutation_rate:
            for node in mutated_individual.values():
                if random.random() < 0.5:  # mutation probability
                    node['value'] += random.uniform(-1.0, 1.0)
                    if node['value'] < node['lower']:
                        node['value'] = node['lower']
                    elif node['value'] > node['upper']:
                        node['value'] = node['upper']
        if random.random() < self.crossover_rate:
            other_individual = random.choice(self.population)
            for i in range(self.tree_size):
                if random.random() < 0.5:
                    mutated_individual[i]['value'] = other_individual[i]['value']
        return mutated_individual

    def get_population(self):
        return self.population

# Example usage
def func(x):
    return x[0]**2 + x[1]**2

budget = 100
dim = 2
algorithm = TreeStructuredGeneticAlgorithm(budget, dim)
algorithm()
population = algorithm.get_population()
for individual in population:
    print(individual)