import random
import numpy as np

class TreeStructuredEvolution:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.tree = self._initialize_tree()
        self.fitness_history = []
        self.hyperparams = self._initialize_hyperparams()

    def _initialize_tree(self):
        tree = {}
        for i in range(self.dim):
            tree[i] = {'lower': -5.0, 'upper': 5.0, 'value': random.uniform(-5.0, 5.0)}
        return tree

    def _initialize_hyperparams(self):
        hyperparams = {}
        for key in self.tree:
            hyperparams[key] = {'lower': -5.0, 'upper': 5.0, 'value': random.uniform(-5.0, 5.0)}
        return hyperparams

    def __call__(self, func):
        for _ in range(self.budget):
            self._evaluate_and_mutate_tree(func)

    def _evaluate_and_mutate_tree(self, func):
        fitness = func(self.tree, self.hyperparams)
        self.fitness_history.append(fitness)
        if fitness == 0:
            return  # termination condition

        for node in self.tree.values():
            if random.random() < 0.35:  # mutation probability
                node['value'] += random.uniform(-1.0, 1.0)
                if node['value'] < node['lower']:
                    node['value'] = node['lower']
                elif node['value'] > node['upper']:
                    node['value'] = node['upper']

        if random.random() < 0.35:  # crossover probability
            other_tree = self._initialize_tree()
            other_hyperparams = self._initialize_hyperparams()
            for i in range(self.dim):
                if random.random() < 0.35:
                    other_tree[i]['value'] = self.tree[i]['value']
                    other_hyperparams[i]['value'] = self.hyperparams[i]['value']
            self.tree = other_tree
            self.hyperparams = other_hyperparams

    def get_tree(self):
        return self.tree

    def get_hyperparams(self):
        return self.hyperparams

# Example usage
def func(x, hyperparams):
    return x[0]**2 + x[1]**2 + hyperparams[0]**2 + hyperparams[1]**2

budget = 100
dim = 2
evolution = TreeStructuredEvolution(budget, dim)
evolution()
tree = evolution.get_tree()
hyperparams = evolution.get_hyperparams()
print(tree)
print(hyperparams)