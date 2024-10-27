import random
import numpy as np
from scipy.special import expit

class MetaHyperTreeEvolution:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.tree = self._initialize_tree()
        self.fitness_history = []
        self.meta_model = self._initialize_meta_model()
        self.meta_learning_rate = 0.01
        self.meta_mutation_rate = 0.35

    def _initialize_tree(self):
        tree = {}
        for i in range(self.dim):
            tree[i] = {'lower': -5.0, 'upper': 5.0, 'value': random.uniform(-5.0, 5.0)}
        return tree

    def _initialize_meta_model(self):
        meta_model = {}
        for i in range(self.dim):
            meta_model[i] = {'weight': 0.5, 'bias': 0.0}
        return meta_model

    def _evaluate_and_mutate_tree(self, func):
        fitness = func(self.tree)
        self.fitness_history.append(fitness)
        if fitness == 0:
            return  # termination condition

        for node in self.tree.values():
            if random.random() < self.meta_mutation_rate:  # mutation probability
                node['value'] += random.uniform(-1.0, 1.0)
                if node['value'] < node['lower']:
                    node['value'] = node['lower']
                elif node['value'] > node['upper']:
                    node['value'] = node['upper']

        for i in range(self.dim):
            if random.random() < 0.5:  # crossover probability
                self.meta_model[i]['weight'] += self.meta_learning_rate * (random.random() - 0.5)
                self.meta_model[i]['weight'] = np.clip(self.meta_model[i]['weight'], 0.0, 1.0)
                self.meta_model[i]['bias'] += self.meta_learning_rate * (random.random() - 0.5)
                self.meta_model[i]['bias'] = np.clip(self.meta_model[i]['bias'], 0.0, 1.0)

        for i in range(self.dim):
            self.tree[i]['value'] = self.meta_model[i]['weight'] * self.tree[i]['value'] + self.meta_model[i]['bias']

    def __call__(self, func):
        for _ in range(self.budget):
            self._evaluate_and_mutate_tree(func)

    def get_tree(self):
        return self.tree

# Example usage
def func(x):
    return x[0]**2 + x[1]**2

budget = 100
dim = 2
evolution = MetaHyperTreeEvolution(budget, dim)
evolution()
tree = evolution.get_tree()
print(tree)