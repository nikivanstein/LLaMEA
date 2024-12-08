# Description: Hierarchical clustering-based Noisy Black Box Optimization using Gradient Descent
# Code: 
# ```python
import numpy as np
import matplotlib.pyplot as plt

class NoisyBlackBoxOptimizer:
    def __init__(self, budget, dim, max_iter=1000):
        self.budget = budget
        self.dim = dim
        self.max_iter = max_iter
        self.explore_eviction = False
        self.current_dim = 0
        self.func = None
        self.cluster_labels = None
        self.population = None
        self.explore_count = 0
        self.best_func = None

    def __call__(self, func):
        if self.explore_eviction:
            # Hierarchical clustering to select the best function to optimize
            self.cluster_labels = np.argpartition(func, self.current_dim)[-1]
            self.explore_eviction = False
            self.population = [func] * self.dim
        else:
            # Perform gradient descent with hierarchical clustering for efficient exploration-ejection
            while self.budget > 0 and self.current_dim < self.dim:
                if self.explore_eviction:
                    # Select the best function to optimize using hierarchical clustering
                    selected_func = np.array([func(x) for x in np.random.uniform(-5.0, 5.0, self.dim)])
                else:
                    # Perform gradient descent with hierarchical clustering for efficient exploration-ejection
                    if self.current_dim == 0:
                        # Gradient descent without hierarchical clustering
                        selected_func = np.array([func(x) for x in np.random.uniform(-5.0, 5.0, self.dim)])
                    else:
                        # Hierarchical clustering to select the best function to optimize
                        selected_func = np.array([func(x) for x in np.random.uniform(-5.0, 5.0, self.dim) if self.cluster_labels == self.cluster_labels[self.current_dim]])
                        self.current_dim += 1
                        if self.budget == 0:
                            break
                self.budget -= 1
                self.population = selected_func

        return self.population

    def func(self, x):
        return np.array([func(x) for func in self.func])

# Exception handling for plotting
try:
    plt.ion()
except ModuleNotFoundError:
    plt.ion = True

# One-line description with the main idea
# Hierarchical clustering-based Noisy Black Box Optimization using Gradient Descent
# to efficiently explore and sample the solution space for optimization tasks.