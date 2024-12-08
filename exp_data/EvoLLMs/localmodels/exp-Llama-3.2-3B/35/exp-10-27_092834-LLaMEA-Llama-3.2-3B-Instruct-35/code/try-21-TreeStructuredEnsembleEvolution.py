import random
import numpy as np

class TreeStructuredEnsembleEvolution:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.ensemble = [self._initialize_tree() for _ in range(10)]
        self.fitness_history = []

    def _initialize_tree(self):
        tree = {}
        for i in range(self.dim):
            tree[i] = {'lower': -5.0, 'upper': 5.0, 'value': random.uniform(-5.0, 5.0)}
        return tree

    def __call__(self, func):
        for _ in range(self.budget):
            self._evaluate_and_evolve(func)

    def _evaluate_and_evolve(self, func):
        fitness = np.mean([func(tree) for tree in self.ensemble])
        self.fitness_history.append(fitness)
        if fitness == 0:
            return  # termination condition

        for tree in self.ensemble:
            if random.random() < 0.35:  # mutation probability
                self._mutate_tree(tree, func)
            if random.random() < 0.35:  # crossover probability
                self._crossover_tree(tree, func)

    def _mutate_tree(self, tree, func):
        for node in tree.values():
            if random.random() < 0.5:  # mutation probability
                node['value'] += random.uniform(-1.0, 1.0)
                if node['value'] < node['lower']:
                    node['value'] = node['lower']
                elif node['value'] > node['upper']:
                    node['value'] = node['upper']

    def _crossover_tree(self, tree, func):
        other_tree = random.choice(self.ensemble)
        for i in range(self.dim):
            if random.random() < 0.5:
                other_tree[i]['value'] = tree[i]['value']
        self.ensemble.append(other_tree)

    def get_ensemble(self):
        return self.ensemble

# Example usage
def func(x):
    return x[0]**2 + x[1]**2

budget = 100
dim = 2
evolution = TreeStructuredEnsembleEvolution(budget, dim)
evolution()
ensemble = evolution.get_ensemble()
print(ensemble)