import random
import numpy as np

class TreeEnsembleEvolution:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.trees = [self._initialize_tree() for _ in range(10)]
        self.fitness_history = []
        self.mutate_probability = 0.35

    def _initialize_tree(self):
        tree = {}
        for i in range(self.dim):
            tree[i] = {'lower': -5.0, 'upper': 5.0, 'value': random.uniform(-5.0, 5.0)}
        return tree

    def __call__(self, func):
        for _ in range(self.budget):
            self._evaluate_and_mutate_trees(func)

    def _evaluate_and_mutate_trees(self, func):
        fitnesses = [func(tree) for tree in self.trees]
        self.fitness_history.append(fitnesses)
        if np.min(fitnesses) == 0:
            return  # termination condition

        for i in range(len(self.trees)):
            tree = self.trees[i]
            if random.random() < self.mutate_probability:
                self._mutate_tree(tree)
            if random.random() < self.mutate_probability:
                self._crossover_tree(tree, random.choice([t for t in self.trees if t!= tree]))

    def _mutate_tree(self, tree):
        for node in tree.values():
            if random.random() < 0.5:
                node['value'] += random.uniform(-1.0, 1.0)
                if node['value'] < node['lower']:
                    node['value'] = node['lower']
                elif node['value'] > node['upper']:
                    node['value'] = node['upper']

    def _crossover_tree(self, tree, other_tree):
        for i in range(self.dim):
            if random.random() < 0.5:
                tree[i]['value'] = other_tree[i]['value']

    def get_trees(self):
        return self.trees

# Example usage
def func(x):
    return x[0]**2 + x[1]**2

budget = 100
dim = 2
evolution = TreeEnsembleEvolution(budget, dim)
evolution()
trees = evolution.get_trees()
for tree in trees:
    print(tree)