import random
import numpy as np

class HyperTreeEvolution:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.tree = self._initialize_tree()
        self.fitness_history = []

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

        if random.random() < 0.35:  # 35% probability to change individual lines
            changed = False
            for i in range(self.dim):
                if random.random() < 0.5:  # mutation probability
                    node = self.tree[i]
                    node['value'] += random.uniform(-1.0, 1.0)
                    if node['value'] < node['lower']:
                        node['value'] = node['lower']
                    elif node['value'] > node['upper']:
                        node['value'] = node['upper']
                    changed = True
            if changed:
                self.tree = self._initialize_tree()

    def get_tree(self):
        return self.tree

# Example usage
def func(x):
    return x[0]**2 + x[1]**2

budget = 100
dim = 2
evolution = HyperTreeEvolution(budget, dim)
evolution()
tree = evolution.get_tree()
print(tree)