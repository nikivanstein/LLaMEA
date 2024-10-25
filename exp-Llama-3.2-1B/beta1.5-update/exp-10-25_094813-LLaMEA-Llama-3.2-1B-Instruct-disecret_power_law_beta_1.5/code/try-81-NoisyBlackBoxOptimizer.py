import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import differential_evolution

class NoisyBlackBoxOptimizer:
    def __init__(self, budget, dim, max_iter=1000):
        self.budget = budget
        self.dim = dim
        self.max_iter = max_iter
        self.explore_eviction = False
        self.current_dim = 0
        self.func = None

    def __call__(self, func):
        if self.explore_eviction:
            # Hierarchical clustering to select the best function to optimize
            cluster_labels = np.argpartition(func, self.current_dim)[-1]
            self.explore_eviction = False
            return func
        else:
            # Gradient descent with hierarchical clustering for efficient exploration-ejection
            while self.budget > 0 and self.current_dim < self.dim:
                if self.explore_eviction:
                    # Select the best function to optimize using hierarchical clustering
                    cluster_labels = np.argpartition(func, self.current_dim)[-1]
                    self.explore_eviction = False
                else:
                    # Perform gradient descent with hierarchical clustering for efficient exploration-ejection
                    if self.current_dim == 0:
                        # Gradient descent without hierarchical clustering
                        self.func = np.array([func(x) for x in np.random.uniform(-5.0, 5.0, self.dim)])
                    else:
                        # Hierarchical clustering to select the best function to optimize
                        cluster_labels = np.argpartition(func, self.current_dim)[-1]
                        self.func = np.array([func(x) for x in np.random.uniform(-5.0, 5.0, self.dim) if cluster_labels == cluster_labels[self.current_dim]])
                        self.current_dim += 1
                        if self.budget == 0:
                            break
                self.budget -= 1
            return self.func

    def func(self, x):
        return np.array([func(x) for func in self.func])

def bbopt(func, budget, dim, max_iter=1000):
    """
    Hierarchical Black Box Optimization using Hierarchical Clustering and Gradient Descent with Hierarchical Clustering for Efficient Exploration-Ejection.

    Args:
        func (function): The black box function to optimize.
        budget (int): The maximum number of function evaluations allowed.
        dim (int): The dimensionality of the search space.
        max_iter (int, optional): The maximum number of iterations. Defaults to 1000.

    Returns:
        function: The optimized function.
    """
    # Initialize the population with random functions
    population = [np.array([func(x) for x in np.random.uniform(-5.0, 5.0, dim)]) for _ in range(100)]

    # Evolve the population using gradient descent with hierarchical clustering
    for _ in range(max_iter):
        # Evaluate the fitness of each individual in the population
        fitness = [self.func(individual) for individual in population]

        # Select the best individuals for the next generation
        selected_indices = np.argsort(fitness)[-self.budget:]

        # Create a new population with the selected individuals
        new_population = [population[i] for i in selected_indices]

        # Update the population with the new individuals
        population = new_population

    return population

# Example usage
budget = 100
dim = 10
optimized_func = bbopt(func, budget, dim)
print(f"Optimized function: {optimized_func}")
print(f"Score: {np.mean(np.abs(optimized_func))}")