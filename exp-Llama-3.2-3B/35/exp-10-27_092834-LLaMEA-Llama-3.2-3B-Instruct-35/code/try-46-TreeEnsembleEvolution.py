import random
import numpy as np

class TreeEnsembleEvolution:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.ensemble = [self._initialize_tree() for _ in range(5)]
        self.fitness_history = []

    def _initialize_tree(self):
        tree = {}
        for i in range(self.dim):
            tree[i] = {'lower': -5.0, 'upper': 5.0, 'value': random.uniform(-5.0, 5.0)}
        return tree

    def __call__(self, func):
        for _ in range(self.budget):
            self._evaluate_and_mutate_ensemble(func)

    def _evaluate_and_mutate_ensemble(self, func):
        fitnesses = [func(tree) for tree in self.ensemble]
        self.fitness_history.append(fitnesses)
        best_tree = self.ensemble[np.argmin(fitnesses)]
        if fitnesses[0] == 0:
            return  # termination condition

        for tree in self.ensemble:
            if random.random() < 0.35:  # mutation probability
                for node in tree.values():
                    node['value'] += random.uniform(-1.0, 1.0)
                    if node['value'] < node['lower']:
                        node['value'] = node['lower']
                    elif node['value'] > node['upper']:
                        node['value'] = node['upper']

        if random.random() < 0.35:  # crossover probability
            new_tree = self._initialize_tree()
            for i in range(self.dim):
                if random.random() < 0.5:
                    new_tree[i]['value'] = best_tree[i]['value']
            self.ensemble[i] = new_tree

    def get_ensemble(self):
        return self.ensemble

# Example usage
def func(x):
    return x[0]**2 + x[1]**2

budget = 100
dim = 2
evolution = TreeEnsembleEvolution(budget, dim)
evolution()
ensemble = evolution.get_ensemble()
print(ensemble)