import random
import numpy as np

class TreeEmbeddedEvolution:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.tree = self._initialize_tree()
        self.fitness_history = []
        self.population = [self.tree]

    def _initialize_tree(self):
        tree = {}
        for i in range(self.dim):
            tree[i] = {'lower': -5.0, 'upper': 5.0, 'value': random.uniform(-5.0, 5.0)}
        return tree

    def __call__(self, func):
        for _ in range(self.budget):
            new_individuals = []
            for individual in self.population:
                new_individual = self._mutate_and_crossover(individual)
                new_individuals.append(new_individual)
            self.population = new_individuals
            self.fitness_history.append(self._evaluate_fitness(func))

    def _mutate_and_crossover(self, individual):
        new_individual = individual.copy()
        for i in range(self.dim):
            if random.random() < 0.35:
                new_individual[i]['value'] += random.uniform(-1.0, 1.0)
                if new_individual[i]['value'] < new_individual[i]['lower']:
                    new_individual[i]['value'] = new_individual[i]['lower']
                elif new_individual[i]['value'] > new_individual[i]['upper']:
                    new_individual[i]['value'] = new_individual[i]['upper']
            if random.random() < 0.35:
                other_individual = random.choice(self.population)
                for j in range(self.dim):
                    if random.random() < 0.5:
                        new_individual[j]['value'] = other_individual[j]['value']
        return new_individual

    def _evaluate_fitness(self, func):
        fitness = []
        for individual in self.population:
            fitness.append(func(individual))
        return np.mean(fitness)

# Example usage
def func(x):
    return x[0]**2 + x[1]**2

budget = 100
dim = 2
evolution = TreeEmbeddedEvolution(budget, dim)
evolution()
tree = evolution.population[0]
print(tree)