import random
import numpy as np

class TreeGraphEvolution:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.tree = self._initialize_tree()
        self.graph = self._initialize_graph()
        self.fitness_history = []

    def _initialize_tree(self):
        tree = {}
        for i in range(self.dim):
            tree[i] = {'lower': -5.0, 'upper': 5.0, 'value': random.uniform(-5.0, 5.0)}
        return tree

    def _initialize_graph(self):
        graph = {}
        for i in range(self.dim):
            graph[i] = {'neighbors': set()}
        return graph

    def __call__(self, func):
        for _ in range(self.budget):
            self._evaluate_and_mutate_tree(func)

    def _evaluate_and_mutate_tree(self, func):
        fitness = func(self.tree)
        self.fitness_history.append(fitness)
        if fitness == 0:
            return  # termination condition

        for node in self.tree.values():
            if random.random() < 0.35:  # mutation probability
                node['value'] += random.uniform(-1.0, 1.0)
                if node['value'] < node['lower']:
                    node['value'] = node['lower']
                elif node['value'] > node['upper']:
                    node['value'] = node['upper']

        if random.random() < 0.35:  # crossover probability
            other_tree = self._initialize_tree()
            for i in range(self.dim):
                if random.random() < 0.35:
                    other_tree[i]['value'] = self.tree[i]['value']
            self.tree = other_tree

        if random.random() < 0.35:  # graph mutation probability
            node = random.choice(list(self.tree.values()))
            neighbor = random.choice(list(self.graph[node['id']]['neighbors']))
            self.graph[neighbor]['neighbors'].add(node['id'])
            self.tree[neighbor]['value'] = self.tree[node['id']]['value']

        if random.random() < 0.35:  # graph crossover probability
            node = random.choice(list(self.tree.values()))
            neighbor = random.choice(list(self.graph[node['id']]['neighbors']))
            self.graph[node['id']]['neighbors'].remove(neighbor)
            self.graph[neighbor]['neighbors'].add(node['id'])
            self.tree[node['id']]['value'] = self.tree[neighbor]['value']

    def get_tree(self):
        return self.tree

    def get_graph(self):
        return self.graph

# Example usage
def func(x):
    return x[0]**2 + x[1]**2

budget = 100
dim = 2
evolution = TreeGraphEvolution(budget, dim)
evolution()
tree = evolution.get_tree()
print(tree)
graph = evolution.get_graph()
print(graph)