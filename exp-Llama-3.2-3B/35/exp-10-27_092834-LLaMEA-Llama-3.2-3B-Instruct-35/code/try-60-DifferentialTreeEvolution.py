import random
import numpy as np

class DifferentialTreeEvolution:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.tree = self._initialize_tree()
        self.fitness_history = []
        self.differential_evolution_params = {'F': 0.5, 'CR': 0.5}

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

        if random.random() < 0.5:  # crossover probability
            other_tree = self._initialize_tree()
            for i in range(self.dim):
                if random.random() < 0.5:
                    other_tree[i]['value'] = self.tree[i]['value']
            self.tree = other_tree

        # Differential Evolution
        if random.random() < self.differential_evolution_params['CR']:
            parents = []
            for _ in range(self.dim):
                parent = self.tree.copy()
                for i in range(self.dim):
                    if random.random() < self.differential_evolution_params['F']:
                        parent[i]['value'] = self.tree[i]['value'] + random.uniform(-1.0, 1.0)
                        if parent[i]['value'] < parent[i]['lower']:
                            parent[i]['value'] = parent[i]['lower']
                        elif parent[i]['value'] > parent[i]['upper']:
                            parent[i]['value'] = parent[i]['upper']
                parents.append(parent)

            # Select best parent
            best_parent = min(parents, key=lambda x: func(x))

            # Update tree with best parent
            for i in range(self.dim):
                self.tree[i]['value'] = best_parent[i]['value']

    def get_tree(self):
        return self.tree

# Example usage
def func(x):
    return x[0]**2 + x[1]**2

budget = 100
dim = 2
evolution = DifferentialTreeEvolution(budget, dim)
evolution()
tree = evolution.get_tree()
print(tree)