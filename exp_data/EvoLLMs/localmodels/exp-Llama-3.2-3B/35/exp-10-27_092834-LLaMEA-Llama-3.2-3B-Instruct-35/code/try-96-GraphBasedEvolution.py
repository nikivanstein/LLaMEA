import random
import numpy as np

class GraphBasedEvolution:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.graph = self._initialize_graph()
        self.fitness_history = []

    def _initialize_graph(self):
        graph = {}
        for i in range(self.dim):
            graph[i] = {'lower': -5.0, 'upper': 5.0, 'value': random.uniform(-5.0, 5.0)}
        return graph

    def __call__(self, func):
        for _ in range(self.budget):
            self._evaluate_and_mutate_graph(func)

    def _evaluate_and_mutate_graph(self, func):
        fitness = func(self.graph)
        self.fitness_history.append(fitness)
        if fitness == 0:
            return  # termination condition

        for node in self.graph.values():
            if random.random() < 0.5:  # mutation probability
                node['value'] += random.uniform(-1.0, 1.0)
                if node['value'] < node['lower']:
                    node['value'] = node['lower']
                elif node['value'] > node['upper']:
                    node['value'] = node['upper']

        if random.random() < 0.5:  # crossover probability
            other_graph = self._initialize_graph()
            for i in range(self.dim):
                if random.random() < 0.5:
                    other_graph[i]['value'] = self.graph[i]['value']
            self.graph = other_graph

    def get_graph(self):
        return self.graph

# Example usage
def func(x):
    return x[0]**2 + x[1]**2

budget = 100
dim = 2
evolution = GraphBasedEvolution(budget, dim)
evolution()
graph = evolution.get_graph()
print(graph)