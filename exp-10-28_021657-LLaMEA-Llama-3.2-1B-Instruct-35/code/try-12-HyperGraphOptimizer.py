# Description: HyperGraph Optimizer with adaptive step size control and probability of exploration
# Code: 
import numpy as np

class HyperGraphOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.graph = self._build_graph()
        self.explore_prob = 0.35
        self.explore_step_size = 0.01

    def _build_graph(self):
        # Create a graph where each node represents a dimension and each edge represents a possible hyper-parameter combination
        graph = {}
        for i in range(self.dim):
            for j in range(i + 1, self.dim):
                graph[(i, j)] = 1
        return graph

    def __call__(self, func):
        # Optimize the black box function using the Hyper-Graph Optimizer
        while self.budget > 0:
            # Generate all possible hyper-parameter combinations
            combinations = self._generate_combinations()
            # Evaluate the function for each combination
            results = [func(combination) for combination in combinations]
            # Find the combination that maximizes the function value
            max_index = np.argmax(results)
            # Update the graph and the optimized function
            self.graph[max_index] = max_index
            # Calculate the step size for the current combination
            step_size = np.exp(-self.explore_step_size * self.explore_prob)
            # Update the budget and explore
            self.budget -= step_size
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

# Example usage:
func = lambda x: x**2
optimizer = HyperGraphOptimizer(100, 10)
optimized_func, optimized_graph = optimizer(func)
print("Optimized function:", optimized_func)
print("Optimized graph:", optimized_graph)