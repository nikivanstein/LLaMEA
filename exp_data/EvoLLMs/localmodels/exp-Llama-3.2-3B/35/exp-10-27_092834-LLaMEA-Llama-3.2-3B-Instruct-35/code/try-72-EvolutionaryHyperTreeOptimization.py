import random
import numpy as np

class EvolutionaryHyperTreeOptimization:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.tree = self._initialize_tree()
        self.fitness_history = []
        self.population_size = 100
        self.mutation_rate = 0.35
        self.crossover_rate = 0.5

    def _initialize_tree(self):
        tree = {}
        for i in range(self.dim):
            tree[i] = {'lower': -5.0, 'upper': 5.0, 'value': random.uniform(-5.0, 5.0)}
        return tree

    def __call__(self, func):
        for _ in range(self.budget):
            self._evaluate_and_mutate_tree(func)

    def _evaluate_and_mutate_tree(self, func):
        fitness = func(self.tree)
        self.fitness_history.append(fitness)
        if fitness == 0:
            return  # termination condition

        new_tree = self.tree.copy()
        for _ in range(self.population_size):
            individual = new_tree.copy()
            if random.random() < self.mutation_rate:
                for i in range(self.dim):
                    if random.random() < 0.5:
                        individual[i]['value'] += random.uniform(-1.0, 1.0)
                        if individual[i]['value'] < individual[i]['lower']:
                            individual[i]['value'] = individual[i]['lower']
                        elif individual[i]['value'] > individual[i]['upper']:
                            individual[i]['value'] = individual[i]['upper']
            if random.random() < self.crossover_rate:
                other_tree = self._initialize_tree()
                for i in range(self.dim):
                    if random.random() < 0.5:
                        other_tree[i]['value'] = individual[i]['value']
                individual = other_tree
            self.tree = individual

    def get_tree(self):
        return self.tree

# Example usage
def func(x):
    return x[0]**2 + x[1]**2

budget = 100
dim = 2
evolution = EvolutionaryHyperTreeOptimization(budget, dim)
evolution()
tree = evolution.get_tree()
print(tree)
