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
        bounds = [(self.graph[i], self.graph[i] + 5) for i in range(self.dim)]
        res = differential_evolution(lambda x: -func(x[0], x[1]), bounds, args=(func,))
        # Update the graph and the optimized function
        self.graph[res.x] = res.x
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

# One-line description with the main idea
# Novel HyperGraph Optimizer for Black Box Optimization
# Refines the strategy by introducing a probabilistic approach to the search space exploration