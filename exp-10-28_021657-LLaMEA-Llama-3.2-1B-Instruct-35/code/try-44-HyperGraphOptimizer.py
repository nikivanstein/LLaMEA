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
            # Refine the strategy
            self.budget, self.graph = self._refine_strategy(self.budget, self.graph, func)
        # Return the optimized function
        return self.budget, self.graph

    def _generate_combinations(self):
        # Generate all possible hyper-parameter combinations
        combinations = []
        for i in range(self.dim):
            for j in range(i + 1, self.dim):
                combinations.append((i, j))
        return combinations

    def _refine_strategy(self, budget, graph, func):
        # Refine the strategy
        while budget > 0 and graph:
            # Find the combination that maximizes the function value
            max_index = np.argmax([func(combination) for combination in graph.values()])
            # Update the graph
            graph[max_index] = max_index
            # Decrement the budget
            budget -= 1
            # If the budget is exhausted, break the loop
            if budget == 0:
                break
        # Return the optimized function and the updated graph
        return budget, graph

# One-line description with the main idea
# HyperGraphOptimizer: A novel metaheuristic algorithm that optimizes black box functions using a hyper-graph structure and a refined strategy to improve convergence.