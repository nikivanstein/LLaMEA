import random
import numpy as np

class EvolutionaryTreeGraphHybrid:
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
            graph[i] = set()
        return graph

    def __call__(self, func):
        for _ in range(self.budget):
            self._evaluate_and_mutate_tree(func)
            self._evaluate_and_mutate_graph(func)

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

    def _evaluate_and_mutate_graph(self, func):
        graph = self.graph
        for node in graph.keys():
            neighbors = list(graph[node])
            if random.random() < 0.5:  # mutation probability
                graph[node].add(random.choice(neighbors))
            if random.random() < 0.5:  # crossover probability
                other_graph = self._initialize_graph()
                for neighbor in graph[node]:
                    if random.random() < 0.5:
                        other_graph[neighbor].add(node)
                self.graph = other_graph

    def get_tree(self):
        return self.tree

    def get_graph(self):
        return self.graph

# Example usage
def func(x):
    return x[0]**2 + x[1]**2

budget = 100
dim = 2
evolution = EvolutionaryTreeGraphHybrid(budget, dim)
evolution()
tree = evolution.get_tree()
print(tree)
graph = evolution.get_graph()
print(graph)