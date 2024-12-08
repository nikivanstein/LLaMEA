import random
import numpy as np
import networkx as nx

class GraphBasedHyperTreeEvolution:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.tree = self._initialize_tree()
        self.graph = nx.Graph()
        self.graph.add_node('root')
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

        # Select a random node to mutate
        node = random.choice(list(self.graph.nodes))
        if random.random() < 0.35:  # mutation probability
            # Randomly select a child node
            child_node = random.choice(list(self.graph.neighbors(node)))
            # Swap the values of the child node with the current node
            self.tree[child_node]['value'], self.tree[node]['value'] = self.tree[node]['value'], self.tree[child_node]['value']
            # Update the graph
            self.graph.nodes[child_node]['value'] = self.tree[child_node]['value']
            self.graph.nodes[node]['value'] = self.tree[node]['value']

        # Perform crossover
        if random.random() < 0.35:  # crossover probability
            # Select two random nodes
            node1 = random.choice(list(self.graph.nodes))
            node2 = random.choice(list(self.graph.neighbors(node1)))
            # Swap the values of the two nodes
            self.tree[node1]['value'], self.tree[node2]['value'] = self.tree[node2]['value'], self.tree[node1]['value']
            # Update the graph
            self.graph.nodes[node1]['value'] = self.tree[node1]['value']
            self.graph.nodes[node2]['value'] = self.tree[node2]['value']

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