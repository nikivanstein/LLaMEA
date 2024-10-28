import numpy as np
from scipy.optimize import minimize
import random

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

    def __call__(self, func, num_evals):
        # Optimize the black box function using the Hyper-Graph Optimizer
        for _ in range(num_evals):
            # Generate all possible hyper-parameter combinations
            combinations = self._generate_combinations()
            # Evaluate the function for each combination
            results = [func(combination, random.uniform(-5.0, 5.0)) for combination in combinations]
            # Find the combination that maximizes the function value
            max_index = np.argmax(results)
            # Update the graph and the optimized function
            self.graph[max_index] = max_index
            # Update the budget
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

# One-line description: HyperGraphOptimizer: A novel metaheuristic algorithm for black box optimization that leverages graph theory to optimize function evaluations.
# Code: 