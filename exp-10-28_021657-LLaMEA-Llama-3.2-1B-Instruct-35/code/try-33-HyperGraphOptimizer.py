# Description: HyperGraphOptimizer: A novel metaheuristic algorithm for solving black box optimization problems.
# Code: 
# ```python
import numpy as np

class HyperGraphOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.graph = self._build_graph()
        self.population = []
        self.population_history = []

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

    def _evaluate(self, func, func_evals):
        # Evaluate the function for a given number of function evaluations
        return np.mean(np.array([func(combination) for combination in combinations]) > func_evals)

    def __call__(self, func, func_evals):
        # Initialize the population with random hyper-parameter combinations
        population = np.random.choice(self._generate_combinations(), size=self.budget, replace=False)
        # Evaluate the initial population
        population_evals = [self._evaluate(func, func_evals) for func, func_evals in zip(func, population)]
        # Create a history of the population evaluations
        self.population_history = population_evals
        # Repeat the process until the budget is exhausted
        while self.budget > 0:
            # Select the fittest individuals
            fittest = np.argsort(self.population_evals)[-self.budget:]
            # Select the fittest individuals for the next generation
            next_generation = np.random.choice(fittest, size=self.budget, replace=False)
            # Create a new population by combining the fittest individuals
            next_population = np.concatenate((population[fittest], next_generation))
            # Evaluate the new population
            new_population_evals = [self._evaluate(func, func_evals) for func, func_evals in zip(func, next_population)]
            # Update the population and the history
            population = next_population
            population_evals = new_population_evals
            # Check for convergence
            if np.allclose(self.population_evals, self.population_history[-1]):
                break
            # Update the graph and the budget
            self.graph = self._build_graph()
            self.budget -= 1
        # Return the optimized function and the best individual
        return self.budget, self.graph[np.argmax(self.population_evals)]

# Description: HyperGraphOptimizer: A novel metaheuristic algorithm for solving black box optimization problems.
# Code: 
# ```python
def optimize_black_box(func, budget, dim):
    optimizer = HyperGraphOptimizer(budget, dim)
    return optimizer.__call__(func, budget)

# Example usage:
func = lambda x: x**2
budget = 100
dim = 2
best_individual, best_function = optimize_black_box(func, budget, dim)
print(f"Best individual: {best_individual}")
print(f"Best function: {best_function}")