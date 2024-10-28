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
        for _ in range(self.budget):
            # Generate all possible hyper-parameter combinations
            combinations = self._generate_combinations()
            # Evaluate the function for each combination
            results = [func(combination) for combination in combinations]
            # Find the combination that maximizes the function value
            max_index = np.argmax(results)
            # Update the graph and the optimized function
            self.graph[max_index] = max_index
            # Refine the search space using a weighted average
            weights = np.array([1.0 / self.dim, 1.0 / (self.dim - 1), 1.0 / (self.dim + 1)])
            self.graph[max_index] = np.min(self.graph.values(), axis=0, keepdims=True)
            self.graph[max_index] = np.clip(self.graph[max_index], -5.0, 5.0)
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

# One-line description: Refine the search space using a weighted average of neighboring nodes in the Hyper-Graph Optimizer.
# Code: 
# ```python
# HyperGraphOptimizer(budget, dim)
# 
# def __call__(self, func):
#     # Optimize the black box function using the Hyper-Graph Optimizer
#     for _ in range(self.budget):
#         # Generate all possible hyper-parameter combinations
#         combinations = self._generate_combinations()
#         # Evaluate the function for each combination
#         results = [func(combination) for combination in combinations]
#         # Find the combination that maximizes the function value
#         max_index = np.argmax(results)
#         # Update the graph and the optimized function
#         self.graph[max_index] = max_index
#         # Refine the search space using a weighted average of neighboring nodes
#         weights = np.array([1.0 / self.dim, 1.0 / (self.dim - 1), 1.0 / (self.dim + 1)])
#         self.graph[max_index] = np.min(self.graph.values(), axis=0, keepdims=True)
#         self.graph[max_index] = np.clip(self.graph[max_index], -5.0, 5.0)
#         # If the budget is exhausted, break the loop
#         if self.budget == 0:
#             break
#     # Return the optimized function
#     return self.budget, self.graph