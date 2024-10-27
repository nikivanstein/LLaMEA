import random
import numpy as np
import networkx as nx

class GraphBasedEvolution:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.tree = self._initialize_tree()
        self.g = self._initialize_graph()
        self.fitness_history = []
        self.mutation_probability = 0.5
        self.crossover_probability = 0.5

    def _initialize_tree(self):
        tree = {}
        for i in range(self.dim):
            tree[i] = {'lower': -5.0, 'upper': 5.0, 'value': random.uniform(-5.0, 5.0)}
        return tree

    def _initialize_graph(self):
        g = nx.Graph()
        for i in range(self.dim):
            g.add_node(i)
        return g

    def __call__(self, func):
        for _ in range(self.budget):
            self._evaluate_and_mutate_tree(func)

    def _evaluate_and_mutate_tree(self, func):
        fitness = func(self.tree)
        self.fitness_history.append(fitness)
        if fitness == 0:
            return  # termination condition

        for node in self.tree.values():
            if random.random() < self.mutation_probability:  # mutation probability
                node['value'] += random.uniform(-1.0, 1.0)
                if node['value'] < node['lower']:
                    node['value'] = node['lower']
                elif node['value'] > node['upper']:
                    node['value'] = node['upper']

        if random.random() < self.crossover_probability:  # crossover probability
            other_tree = self._initialize_tree()
            for i in range(self.dim):
                if random.random() < 0.5:
                    other_tree[i]['value'] = self.tree[i]['value']
            self.tree = other_tree

        # Graph mutation
        for i in range(self.dim):
            for j in range(i+1, self.dim):
                if random.random() < 0.5:
                    self.g.add_edge(i, j)
                    self.g.add_edge(j, i)

        # Graph crossover
        if random.random() < 0.5:
            other_tree = self._initialize_tree()
            other_g = self._initialize_graph()
            for i in range(self.dim):
                if random.random() < 0.5:
                    other_tree[i]['value'] = self.tree[i]['value']
            for i in range(self.dim):
                for j in range(i+1, self.dim):
                    if random.random() < 0.5:
                        if self.g.has_edge(i, j):
                            other_g.add_edge(i, j)
                        else:
                            other_g.add_edge(j, i)
            self.tree = other_tree
            self.g = other_g

    def get_tree(self):
        return self.tree

# Example usage
def func(x):
    return x[0]**2 + x[1]**2

budget = 100
dim = 2
evolution = GraphBasedEvolution(budget, dim)
evolution()
tree = evolution.get_tree()
print(tree)