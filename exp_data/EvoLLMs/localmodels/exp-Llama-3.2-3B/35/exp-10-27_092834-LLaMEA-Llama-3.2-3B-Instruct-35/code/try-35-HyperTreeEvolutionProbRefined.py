import random
import numpy as np

class HyperTreeEvolutionProbRefined:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.tree = self._initialize_tree()
        self.fitness_history = []
        self.refinement_prob = 0.35

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

        if random.random() < self.refinement_prob:
            self._refine_tree(func)

    def _refine_tree(self, func):
        # Select a random node to refine
        node_to_refine = random.choice(list(self.tree.values()))
        # Refine the node by changing its value with a probability of 0.5
        if random.random() < 0.5:
            node_to_refine['value'] += random.uniform(-1.0, 1.0)
            if node_to_refine['value'] < node_to_refine['lower']:
                node_to_refine['value'] = node_to_refine['lower']
            elif node_to_refine['value'] > node_to_refine['upper']:
                node_to_refine['value'] = node_to_refine['upper']
        # Refine the children of the node with a probability of 0.2
        if random.random() < 0.2:
            child_to_refine = random.choice(list(node_to_refine.values()))
            child_to_refine['value'] += random.uniform(-1.0, 1.0)
            if child_to_refine['value'] < child_to_refine['lower']:
                child_to_refine['value'] = child_to_refine['lower']
            elif child_to_refine['value'] > child_to_refine['upper']:
                child_to_refine['value'] = child_to_refine['upper']

    def get_tree(self):
        return self.tree

# Example usage
def func(x):
    return x[0]**2 + x[1]**2

budget = 100
dim = 2
evolution = HyperTreeEvolutionProbRefined(budget, dim)
evolution()
tree = evolution.get_tree()
print(tree)