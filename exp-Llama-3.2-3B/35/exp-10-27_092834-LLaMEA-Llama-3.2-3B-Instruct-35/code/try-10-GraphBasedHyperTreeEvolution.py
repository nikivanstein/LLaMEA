import random
import numpy as np
import networkx as nx

class GraphBasedHyperTreeEvolution:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.tree = self._initialize_tree()
        self.graph = nx.Graph()
        self.graph.add_node(0)
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

        if random.random() < 0.5:  # crossover probability
            new_node = self._initialize_tree()
            for i in range(self.dim):
                if random.random() < 0.5:
                    new_node[i]['value'] = self.tree[i]['value']
            self.graph.add_node(len(self.graph), self.tree)
            self.graph.add_edge(0, len(self.graph))
            for i in range(len(self.graph)):
                for j in range(i+1, len(self.graph)):
                    if random.random() < 0.35:
                        self.graph.add_edge(i, j)
            self.tree = new_node

        self.graph.nodes(data=True)
        fitness_values = [func(node['value']) for node in self.graph.nodes(data=True)]
        self.graph.nodes(data={'fitness': fitness_values})
        fitness_values = [node['fitness']['fitness'] for node in self.graph.nodes(data=True)]
        self.graph.nodes(data={'fitness': fitness_values})
        max_fitness = max(fitness_values)
        max_index = fitness_values.index(max_fitness)
        self.graph.nodes(data={'fitness': fitness_values})
        self.graph.nodes(max_index, 'best_node')

    def get_tree(self):
        return self.tree

# Example usage
def func(x):
    return x[0]**2 + x[1]**2

budget = 100
dim = 2
evolution = GraphBasedHyperTreeEvolution(budget, dim)
evolution()
tree = evolution.get_tree()
print(tree)