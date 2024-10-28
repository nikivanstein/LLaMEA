# Description: HyperGraphOptimizer with Adaptive Sampling and Hyper-Parameter Tuning
# Code: 
# ```python
import numpy as np
import random

class HyperGraphOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.graph = self._build_graph()
        self.sampling_rate = 0.1
        self.tuning_rate = 0.05
        self.tuning_budget = 100

    def _build_graph(self):
        # Create a graph where each node represents a dimension and each edge represents a possible hyper-parameter combination
        graph = {}
        for i in range(self.dim):
            for j in range(i + 1, self.dim):
                graph[(i, j)] = 1
        return graph

    def __call__(self, func):
        # Optimize the black box function using the Hyper-Graph Optimizer
        results = []
        for _ in range(self.budget):
            # Generate all possible hyper-parameter combinations
            combinations = self._generate_combinations()
            # Evaluate the function for each combination
            results.append([func(combination) for combination in combinations])
        # Find the combination that maximizes the function value
        max_index = np.argmax(results)
        # Update the graph and the optimized function
        self.graph[max_index] = max_index
        # Perform adaptive sampling to reduce the number of function evaluations
        self.sample_function(results, max_index)
        # Perform hyper-parameter tuning to refine the search space
        self.tune_function(results, max_index)
        # Return the optimized function
        return self.budget, self.graph

    def _generate_combinations(self):
        # Generate all possible hyper-parameter combinations
        combinations = []
        for i in range(self.dim):
            for j in range(i + 1, self.dim):
                combinations.append((i, j))
        return combinations

    def _sample_function(self, results, max_index):
        # Perform adaptive sampling to reduce the number of function evaluations
        if random.random() < self.sampling_rate:
            # Randomly sample a subset of combinations
            combinations = random.sample(results, len(results) // 10)
        else:
            # Evaluate the function for the full set of combinations
            combinations = results
        # Find the combination that maximizes the function value
        max_index = np.argmax(combinations)
        # Update the graph and the optimized function
        self.graph[max_index] = max_index

    def _tune_function(self, results, max_index):
        # Perform hyper-parameter tuning to refine the search space
        tuning_combinations = []
        for i in range(self.dim):
            for j in range(i + 1, self.dim):
                # Evaluate the function at the current combination
                result = results[max_index][i] * results[max_index][j]
                # Update the tuning combinations
                tuning_combinations.append((i, j))
        # Find the combination that maximizes the function value
        max_index = np.argmax(tuning_combinations)
        # Update the graph and the optimized function
        self.graph[max_index] = max_index