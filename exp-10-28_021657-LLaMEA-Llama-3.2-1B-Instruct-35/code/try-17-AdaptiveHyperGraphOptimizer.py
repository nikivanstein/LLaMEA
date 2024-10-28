import numpy as np

class AdaptiveHyperGraphOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.graph = self._build_graph()
        self.best_func = None
        self.best_func_score = 0.0
        self.best_func_idx = None
        self.min_evals = 0
        self.max_evals = 0

    def _build_graph(self):
        # Create a graph where each node represents a dimension and each edge represents a possible hyper-parameter combination
        graph = {}
        for i in range(self.dim):
            for j in range(i + 1, self.dim):
                graph[(i, j)] = 1
        return graph

    def __call__(self, func):
        # Optimize the black box function using the Adaptive HyperGraph Optimizer
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
        # Update the best function and its score
        self.best_func = func
        self.best_func_score = np.max(results)
        self.best_func_idx = max_index
        # If the number of evaluations exceeds the minimum threshold, refine the search strategy
        if self.min_evals < self.max_evals:
            # Calculate the convergence rate
            convergence_rate = np.max(np.abs(results) / self.min_evals) ** 0.5
            # If the convergence rate is above the specified threshold (0.35), refine the search strategy
            if convergence_rate > 0.35:
                # Calculate the average number of evaluations per iteration
                avg_evals_per_iter = np.mean([self.budget, self.min_evals])
                # Refine the search strategy by increasing the number of evaluations per iteration
                self.min_evals *= 1.1
                # Update the best function and its score
                self.best_func_score = np.max(results)
                self.best_func_idx = max_index
                # If the number of evaluations exceeds the maximum threshold, break the loop
                if self.min_evals > self.max_evals:
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
def func1(x):
    return np.sin(x)

def func2(x):
    return np.exp(x)

optimizer = AdaptiveHyperGraphOptimizer(budget=100, dim=2)
best_func, best_graph = optimizer(func1)
print(f"Best function: {best_func.__name__}, Best score: {best_func_score}, Best index: {best_func_idx}")
print(f"Graph: {best_graph}")