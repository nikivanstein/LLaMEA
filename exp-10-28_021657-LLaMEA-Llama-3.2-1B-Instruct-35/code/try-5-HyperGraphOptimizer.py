import numpy as np
import random

class HyperGraphOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.graph = self._build_graph()
        self.search_strategy = self._adaptive_search_strategy()

    def _build_graph(self):
        # Create a graph where each node represents a dimension and each edge represents a possible hyper-parameter combination
        graph = {}
        for i in range(self.dim):
            for j in range(i + 1, self.dim):
                graph[(i, j)] = 1
        return graph

    def _generate_combinations(self):
        # Generate all possible hyper-parameter combinations
        combinations = []
        for i in range(self.dim):
            for j in range(i + 1, self.dim):
                combinations.append((i, j))
        return combinations

    def _adaptive_search_strategy(self):
        # Initialize the search strategy with a random strategy
        strategies = [self._random_strategy]
        for _ in range(10):  # Run for 10 iterations to converge
            # Evaluate the function for each combination
            results = [self._evaluate_func(combination) for combination in self._generate_combinations()]
            # Find the combination that maximizes the function value
            max_index = np.argmax(results)
            # Update the graph and the optimized function
            self.graph[max_index] = max_index
            self.budget -= 1
            # If the budget is exhausted, break the loop
            if self.budget == 0:
                break
            # Update the search strategy based on the performance of the current strategy
            if results[max_index] > results[ strategies[-1][0] ]:
                strategies.append((max_index, results[max_index]))
        return strategies

    def _random_strategy(self, combination):
        # Randomly select a hyper-parameter combination
        return combination

    def _evaluate_func(self, combination):
        # Evaluate the black box function using the current combination
        func = lambda x: self._evaluate_func_func(x, combination)
        return func

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

    def _evaluate_func_func(self, x, combination):
        # Evaluate the function using the current combination and the graph
        func = lambda y: np.dot(y, x)
        return func