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

    def __call__(self, func):
        if self.explore_eviction:
            # Hierarchical clustering to select the best function to optimize
            cluster_labels = np.argpartition(func, self.current_dim)[-1]
            self.explore_eviction = False
            return func
        else:
            # Hierarchical genetic algorithm with evolutionary crossover for efficient exploration-ejection
            while self.budget > 0 and self.current_dim < self.dim:
                if self.explore_eviction:
                    # Select the best function to optimize using hierarchical clustering
                    cluster_labels = np.argpartition(func, self.current_dim)[-1]
                    self.explore_eviction = False
                else:
                    # Perform hierarchical genetic algorithm with evolutionary crossover
                    if self.current_dim == 0:
                        # Hierarchical genetic algorithm without evolutionary crossover
                        self.func = np.array([func(x) for x in np.random.uniform(-5.0, 5.0, self.dim)])
                    else:
                        # Hierarchical genetic algorithm with evolutionary crossover
                        parent1 = np.array([func(x) for x in random.sample(self.func, 2)])
                        parent2 = np.array([func(x) for x in random.sample(self.func, 2)])
                        child = np.mean([parent1, parent2], axis=0)
                        if np.random.rand() < 0.5:
                            child = np.concatenate((child, parent1))
                        else:
                            child = np.concatenate((child, parent2))
                        self.func = np.array([func(x) for x in child])
                        self.current_dim += 1
                        if self.budget == 0:
                            break
                self.budget -= 1
            return self.func

    def func(self, x):
        return np.array([func(x) for func in self.func])

# One-line description with main idea
# Hierarchical Black Box Optimization using Genetic Algorithm with Evolutionary Crossover
# to efficiently explore the search space and find the optimal function
# with a probability of 0.047619047619047616
# 