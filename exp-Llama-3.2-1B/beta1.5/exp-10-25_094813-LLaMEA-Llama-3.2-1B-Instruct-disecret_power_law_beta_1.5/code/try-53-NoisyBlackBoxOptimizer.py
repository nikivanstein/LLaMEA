import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import distance
from scipy.optimize import minimize

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
            # Hierarchical clustering-based genetic algorithm for efficient exploration-ejection
            # with a hierarchical clustering strategy to select the best individual
            # to optimize
            if self.current_dim == 0:
                # Initialize population with random individuals
                self.population = [func(np.random.uniform(-5.0, 5.0, self.dim)) for _ in range(100)]
            else:
                # Select the best individual to optimize using hierarchical clustering
                cluster_labels = np.argpartition(func, self.current_dim)[-1]
                self.population = [func(x) for x in np.random.uniform(-5.0, 5.0, self.dim) if cluster_labels == cluster_labels[self.current_dim]]
                self.current_dim += 1
                if self.budget == 0:
                    # Select the best individual to optimize using hierarchical clustering
                    cluster_labels = np.argpartition(func, self.current_dim)[-1]
                    self.population = [func(x) for x in np.random.uniform(-5.0, 5.0, self.dim) if cluster_labels == cluster_labels[self.current_dim]]
                    self.current_dim -= 1
                    if self.budget == 0:
                        break
            self.budget -= 1
            return self.population

    def func(self, x):
        return np.array([func(x) for func in self.func])

# One-line description with main idea
# Hierarchical Clustering-based Genetic Algorithm for Efficient Exploration-Ejection