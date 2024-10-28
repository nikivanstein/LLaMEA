import numpy as np
from scipy.optimize import differential_evolution

class HyperGraphOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.graph = self._build_graph()

    def _build_graph(self):
        # Create a graph where each node represents a dimension and each edge represents a possible hyper-parameter combination
        graph = {}
        for i in range(self.dim):
            for j in range(i + 1, self.dim):
                graph[(i, j)] = 1
        return graph

    def __call__(self, func):
        # Optimize the black box function using the Hyper-Graph Optimizer
        bounds = [(-5.0, 5.0)] * self.dim
        res = differential_evolution(lambda x: -func(x), [(x, bounds) for x in range(-5.0, 6.0)], x0=[1.0] * self.dim, tol=1e-6)
        # Update the graph and the optimized function
        self.graph[res.x[0], res.x[1]] = res.x[0]
        self.budget -= 1
        # If the budget is exhausted, break the loop
        if self.budget == 0:
            break
        # Return the optimized function
        return self.budget, self.graph

    def _generate_combinations(self):
        # Generate all possible hyper-parameter combinations
        combinations = []
        for i in range(self.dim):
            for j in range(i + 1, self.dim):
                combinations.append((i, j))
        return combinations

# HyperGraph Optimizer with Refined Strategy
# Code: 