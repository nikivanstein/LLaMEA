import random
import numpy as np

class TreeStructuredEnsemble:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.trees = [self._initialize_tree() for _ in range(10)]
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
        fitness = self._evaluate(self.trees, func)
        self.fitness_history.append(fitness)
        if fitness == 0:
            return  # termination condition

        for tree in self.trees:
            if random.random() < 0.35:  # mutation probability
                self._mutate_tree(tree, func)

        if random.random() < 0.35:  # crossover probability
            self._crossover_trees()

    def _evaluate(self, trees, func):
        fitnesses = []
        for tree in trees:
            fitness = func(tree)
            fitnesses.append(fitness)
        return np.mean(fitnesses)

    def _mutate_tree(self, tree, func):
        for node in tree.values():
            if random.random() < 0.5:  # mutation probability
                node['value'] += random.uniform(-1.0, 1.0)
                if node['value'] < node['lower']:
                    node['value'] = node['lower']
                elif node['value'] > node['upper']:
                    node['value'] = node['upper']

    def _crossover_trees(self):
        new_trees = []
        for _ in range(len(self.trees)):
            new_tree = {}
            for i in range(self.dim):
                if random.random() < 0.5:
                    new_tree[i] = self.trees[random.randint(0, len(self.trees)-1)][i]
                else:
                    new_tree[i] = self.trees[random.randint(0, len(self.trees)-1)][i]
            new_trees.append(new_tree)
        self.trees = new_trees

# Example usage
def func(x):
    return x[0]**2 + x[1]**2

budget = 100
dim = 2
ensemble = TreeStructuredEnsemble(budget, dim)
ensemble()
trees = [ensemble.get_tree() for _ in range(10)]
print(trees)