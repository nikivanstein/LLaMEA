# Description: Adaptive HyperGraph Optimizer for Black Box Optimization
# Code: 
# ```python
import numpy as np

class AdaptiveHyperGraphOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.graph = self._build_graph()
        self.best_solution = None
        self.best_score = 0.0
        self.best_combinations = None

    def _build_graph(self):
        # Create a graph where each node represents a dimension and each edge represents a possible hyper-parameter combination
        graph = {}
        for i in range(self.dim):
            for j in range(i + 1, self.dim):
                graph[(i, j)] = 1
        return graph

    def __call__(self, func, iterations=100, alpha=0.35):
        # Optimize the black box function using the Adaptive HyperGraph Optimizer
        for i in range(iterations):
            # Generate all possible hyper-parameter combinations
            combinations = self._generate_combinations()
            # Evaluate the function for each combination
            results = [func(combination) for combination in combinations]
            # Find the combination that maximizes the function value
            max_index = np.argmax(results)
            # Update the graph and the optimized function
            self.graph[max_index] = max_index
            self.budget -= 1
            # If the budget is exhausted, break the loop
            if self.budget == 0:
                break
        # Return the optimized function and the best solution found
        return self.budget, self.graph, self.best_solution, self.best_score, self.best_combinations

    def _generate_combinations(self):
        # Generate all possible hyper-parameter combinations
        combinations = []
        for i in range(self.dim):
            for j in range(i + 1, self.dim):
                combinations.append((i, j))
        return combinations

# One-line description with the main idea
# Adaptive HyperGraph Optimizer for black box optimization, 
# using adaptive strategy to refine the search space based on the individual lines of the selected solution