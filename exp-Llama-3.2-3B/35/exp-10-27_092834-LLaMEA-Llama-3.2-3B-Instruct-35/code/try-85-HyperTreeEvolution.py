import random
import numpy as np

class HyperTreeEvolution:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.tree = self._initialize_tree()
        self.fitness_history = []
        self.refinement_probability = 0.35

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

        for node in self.tree.values():
            if random.random() < 0.5:  # mutation probability
                node['value'] += random.uniform(-1.0, 1.0)
                if node['value'] < node['lower']:
                    node['value'] = node['lower']
                elif node['value'] > node['upper']:
                    node['value'] = node['upper']

        if random.random() < self.refinement_probability:  # refinement probability
            self._refine_tree(func)

    def _refine_tree(self, func):
        for node in self.tree.values():
            if random.random() < 0.5:  # refinement probability
                new_value = node['value'] + random.uniform(-0.1, 0.1)
                if new_value < node['lower']:
                    new_value = node['lower']
                elif new_value > node['upper']:
                    new_value = node['upper']
                node['value'] = new_value
        self.tree = self._prune_tree(func)

    def _prune_tree(self, func):
        new_tree = {}
        for node in self.tree.values():
            if node['value'] == self.tree[self.tree.keys()[0]]['value']:
                new_tree[self.tree.keys()[0]] = node
            else:
                new_tree[node['key']] = {'lower': node['lower'], 'upper': node['upper'], 'value': node['value']}
        return new_tree

# Example usage
def func(x):
    return x[0]**2 + x[1]**2

budget = 100
dim = 2
evolution = HyperTreeEvolution(budget, dim)
evolution()
tree = evolution.get_tree()
print(tree)