import numpy as np
import random

class AdaptiveHyperGraphOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.graph = self._build_graph()
        self.search_strategy = self._init_search_strategy()

    def _build_graph(self):
        # Create a graph where each node represents a dimension and each edge represents a possible hyper-parameter combination
        graph = {}
        for i in range(self.dim):
            for j in range(i + 1, self.dim):
                graph[(i, j)] = 1
        return graph

    def _init_search_strategy(self):
        # Initialize the search strategy with a random starting point
        return random.randint(0, self.dim - 1), random.randint(0, self.dim - 1)

    def __call__(self, func):
        # Optimize the black box function using the Adaptive Hyper-Graph Optimizer
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
        return combinations

    def update_search_strategy(self):
        # Update the search strategy based on the convergence rate
        if self.budget == 0:
            return self.search_strategy
        else:
            # Calculate the average number of evaluations required to reach the budget
            avg_evaluations = self.budget / np.log(2)
            # Update the search strategy to prioritize evaluations with higher convergence rates
            if avg_evaluations < 0.35:
                return (self.search_strategy[0] + 1, self.search_strategy[1])
            else:
                return self.search_strategy

    def __str__(self):
        # Return a string representation of the optimizer
        return f"AdaptiveHyperGraphOptimizer(budget={self.budget}, dim={self.dim})"

# Description: Adaptive Hyper-Graph Optimizer with Adaptive Search Strategy
# Code: 
# ```python
optimizer = AdaptiveHyperGraphOptimizer(100, 5)
print(optimizer)