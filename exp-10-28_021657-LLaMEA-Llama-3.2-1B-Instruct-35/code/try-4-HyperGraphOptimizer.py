# Description: HyperGraphOptimizer with Adaptive Sampling Strategy
# Code: 
# ```python
import numpy as np

class HyperGraphOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.graph = self._build_graph()
        self.sample_size = self.budget / 10
        self.current_sample = 0

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
        # Sample combinations based on the current sample size
        if self.current_sample < self.sample_size:
            combinations = np.random.choice(len(combinations), size=self.sample_size, replace=False)
        else:
            # Use the current sample to select combinations
            combinations = np.random.choice(len(combinations), size=self.budget, replace=False)
        return combinations

    def _sample_hyper_parameters(self):
        # Sample hyper-parameter combinations from the current sample
        self.current_sample = np.random.randint(0, self.budget)
        combinations = np.random.choice(len(self.graph), size=self.current_sample, replace=False)
        return combinations

    def _evaluate_function(self, func, combinations):
        # Evaluate the function for the selected combinations
        results = [func(combination) for combination in combinations]
        # Find the combination that maximizes the function value
        max_index = np.argmax(results)
        return max_index

# One-line description: HyperGraphOptimizer with Adaptive Sampling Strategy
# Code: 
# ```python
hyper_graph_optimizer = HyperGraphOptimizer(budget=100, dim=10)
hyper_graph_optimizer.__call__(func=lambda x: x**2)
print(hyper_graph_optimizer.budget, hyper_graph_optimizer.graph)