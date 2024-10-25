import numpy as np
import matplotlib.pyplot as plt
import random

class NoisyBlackBoxOptimizer:
    def __init__(self, budget, dim, max_iter=1000):
        self.budget = budget
        self.dim = dim
        self.max_iter = max_iter
        self.explore_eviction = False
        self.current_dim = 0
        self.func = None
        self.mutation_rate = 0.1
        self.mutation_threshold = 0.05

    def __call__(self, func):
        if self.explore_eviction:
            # Hierarchical clustering to select the best function to optimize
            cluster_labels = np.argpartition(func, self.current_dim)[-1]
            self.explore_eviction = False
            return func
        else:
            # Perform gradient descent with hierarchical clustering for efficient exploration-ejection
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
                # Update the fitness of the best individual
                best_individual = self.func[np.argmax(self.func)]
                self.func = np.array([func(x) for x in np.random.uniform(-5.0, 5.0, self.dim) if x!= best_individual])
                if np.linalg.norm(self.func - best_individual) < self.mutation_threshold:
                    self.func = self.func + np.random.uniform(-1.0, 1.0, self.dim)

    def func(self, x):
        return np.array([func(x) for func in self.func])

# One-line description: Hierarchical clustering with adaptive mutation to optimize black box functions.
# Code: 