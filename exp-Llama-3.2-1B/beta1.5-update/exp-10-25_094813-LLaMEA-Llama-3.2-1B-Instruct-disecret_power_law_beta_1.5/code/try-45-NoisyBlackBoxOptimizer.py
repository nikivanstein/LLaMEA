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
            # Hierarchical clustering-based Noisy Black Box Optimization Algorithm
            # Use differential evolution to optimize the black box function
            bounds = [(-5.0, 5.0) for _ in range(self.dim)]
            res = differential_evolution(lambda x: -np.array([func(x[i]) for i in range(self.dim)]), bounds, args=(func,), bounds=bounds, x0=np.array([func(np.random.uniform(-5.0, 5.0, self.dim)) for _ in range(self.dim)]), n_iter=self.max_iter)
            if res.fun > 0:
                self.func = np.array([func(x) for x in res.x])
            else:
                self.func = np.array([func(x) for x in np.random.uniform(-5.0, 5.0, self.dim)])
            return self.func

    def func(self, x):
        return np.array([func(x) for func in self.func])

# One-line description: Hierarchical Clustering-based Noisy Black Box Optimization Algorithm