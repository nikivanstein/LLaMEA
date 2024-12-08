import random
import numpy as np

class TreeStructuredHyperparameterOptimization:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.tree = self._initialize_tree()
        self.fitness_history = []
        self.search_space = [(-5.0, 5.0)] * self.dim

    def _initialize_tree(self):
        tree = {}
        for i in range(self.dim):
            tree[i] = {'lower': self.search_space[i][0], 'upper': self.search_space[i][1], 'value': random.uniform(self.search_space[i][0], self.search_space[i][1]), 'children': []}
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

            if random.random() < 0.5:  # crossover probability
                other_node = random.choice(list(self.tree.values()))
                if node!= other_node:
                    other_node['value'] = node['value']
                    node['children'].append(other_node)

        # Probability-based line search
        if random.random() < 0.35:
            for node in self.tree.values():
                step = random.uniform(-1.0, 1.0)
                new_value = node['value'] + step
                if new_value < node['lower']:
                    new_value = node['lower']
                elif new_value > node['upper']:
                    new_value = node['upper']
                node['value'] = new_value

    def get_tree(self):
        return self.tree

# Example usage
def func(x):
    return x[0]**2 + x[1]**2

budget = 100
dim = 2
optimization = TreeStructuredHyperparameterOptimization(budget, dim)
optimization()
tree = optimization.get_tree()
print(tree)