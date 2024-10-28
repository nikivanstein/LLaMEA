# Description: Adaptive Hyper-Graph Optimizer with Boundary Exploration
# Code: 
# ```python
import numpy as np

class AdaptiveHyperGraphOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.graph = self._build_graph()
        self.explore_count = 0
        self.explore_threshold = 0.35

    def _build_graph(self):
        # Create a graph where each node represents a dimension and each edge represents a possible hyper-parameter combination
        graph = {}
        for i in range(self.dim):
            for j in range(i + 1, self.dim):
                graph[(i, j)] = 1
        return graph

    def __call__(self, func):
        # Optimize the black box function using the Adaptive Hyper-Graph Optimizer
        while self.budget > 0:
            # Generate all possible hyper-parameter combinations
            combinations = self._generate_combinations()
            # Evaluate the function for each combination
            results = [func(combination) for combination in combinations]
            # Find the combination that maximizes the function value
            max_index = np.argmax(results)
            # Update the graph and the optimized function
            self.graph[max_index] = max_index
            # Explore the graph
            if self.explore_count < self.explore_threshold * self.budget:
                # Get a random subset of hyper-parameter combinations
                subsets = self._get_random_subsets()
                # Evaluate the function for each subset
                results = [func(subset) for subset in subsets]
                # Find the subset that maximizes the function value
                max_subset_index = np.argmax(results)
                # Update the graph and the optimized function
                self.graph[max_subset_index] = max_subset_index
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

    def _get_random_subsets(self):
        # Get a random subset of hyper-parameter combinations
        subsets = []
        for _ in range(int(self.explore_threshold * self.budget)):
            subset = np.random.choice(len(self.graph), size=self.dim, replace=False)
            subsets.append(subset)
        return subsets

    def adapt(self):
        # Refine the strategy by changing the individual lines of the selected solution
        # Increase the explore threshold to 0.4 and decrease the budget to 0.2
        self.explore_threshold = 0.4
        self.budget = 0.2

# Create an instance of the AdaptiveHyperGraphOptimizer
optimizer = AdaptiveHyperGraphOptimizer(100, 5)

# Optimize the black box function using the Adaptive Hyper-Graph Optimizer
optimizer(optimized_func)