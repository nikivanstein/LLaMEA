import numpy as np
from scipy.optimize import differential_evolution

class HyperGraphOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.graph = self._build_graph()
        self.x = None

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
        return combinations

    def optimize_function(self, func):
        # Optimize the black box function using the Hyper-Graph Optimizer
        # with a probability of 0.35 refinement
        if np.random.rand() < 0.35:
            # Refine the search space
            self.x = np.array([i / self.dim for i in range(self.dim)])
        else:
            # Use the original search space
            self.x = np.array([i for i in range(self.dim)])

        # Optimize the function
        result = differential_evolution(lambda x: -func(x), self.graph.items(), x0=self.x)
        return result.x, result.fun

# Example usage
def func(x):
    return np.sum(x**2)

optimizer = HyperGraphOptimizer(budget=100, dim=10)
optimized_func, optimized_budget, optimized_graph = optimizer.optimize_function(func)
print("Optimized function:", optimized_func)
print("Optimized budget:", optimized_budget)
print("Optimized graph:", optimized_graph)