import random
import numpy as np

class HybridEvolutionaryAlgorithm:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.tree = self._initialize_tree()
        self.population = self._initialize_population()
        self.fitness_history = []

    def _initialize_tree(self):
        tree = {}
        for i in range(self.dim):
            tree[i] = {'lower': -5.0, 'upper': 5.0, 'value': random.uniform(-5.0, 5.0)}
        return tree

    def _initialize_population(self):
        population = []
        for _ in range(100):
            individual = {}
            for i in range(self.dim):
                individual[i] = {'lower': -5.0, 'upper': 5.0, 'value': random.uniform(-5.0, 5.0)}
            population.append(individual)
        return population

    def __call__(self, func):
        for _ in range(self.budget):
            self._evaluate_and_mutate_population(func)

    def _evaluate_and_mutate_population(self, func):
        fitness = []
        for individual in self.population:
            fitness.append(func(individual))
        self.fitness_history.append(fitness)

        if min(fitness) == max(fitness):
            return  # termination condition

        for individual in self.population:
            if random.random() < 0.35:
                self._mutate_individual(individual, func)

        if random.random() < 0.35:
            self._crossover_population(func)

    def _mutate_individual(self, individual, func):
        for i in range(self.dim):
            if random.random() < 0.5:  # mutation probability
                individual[i]['value'] += random.uniform(-1.0, 1.0)
                if individual[i]['value'] < individual[i]['lower']:
                    individual[i]['value'] = individual[i]['lower']
                elif individual[i]['value'] > individual[i]['upper']:
                    individual[i]['value'] = individual[i]['upper']

    def _crossover_population(self, func):
        new_population = []
        for _ in range(100):
            parent1 = random.choice(self.population)
            parent2 = random.choice(self.population)
            child = {}
            for i in range(self.dim):
                if random.random() < 0.5:
                    child[i] = {'lower': parent1[i]['lower'], 'upper': parent2[i]['upper'], 'value': (parent1[i]['value'] + parent2[i]['value']) / 2}
            new_population.append(child)
        self.population = new_population

    def get_tree(self):
        return self.tree

    def get_population(self):
        return self.population

# Example usage
def func(x):
    return x[0]**2 + x[1]**2

budget = 100
dim = 2
evolution = HybridEvolutionaryAlgorithm(budget, dim)
evolution()
tree = evolution.get_tree()
population = evolution.get_population()
print(tree)
print(population)