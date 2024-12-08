import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import distance

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
            # Hierarchical clustering-based Noisy Black Box Optimization
            while self.budget > 0 and self.current_dim < self.dim:
                if self.explore_eviction:
                    # Select the best function to optimize using hierarchical clustering
                    cluster_labels = np.argpartition(func, self.current_dim)[-1]
                    self.explore_eviction = False
                else:
                    # Perform hierarchical clustering to refine the strategy
                    cluster_labels = np.argpartition(func, self.current_dim)[-1]
                    self.func = np.array([func(x) for x in np.random.uniform(-5.0, 5.0, self.dim) if cluster_labels == cluster_labels[self.current_dim]])
                    self.current_dim += 1
                    if self.budget == 0:
                        break
                self.budget -= 1
            return self.func

    def func(self, x):
        return np.array([func(x) for func in self.func])

# Example usage:
# Create a noisy black box function
def func(x):
    return np.random.normal(0, 1, x.shape)

# Initialize the optimizer
optimizer = NoisyBlackBoxOptimizer(budget=100, dim=10, max_iter=100)

# Optimize the function using the optimizer
new_individual = optimizer.func(np.random.uniform(-10, 10, 10))

# Plot the function and the new individual
plt.plot(func(np.random.uniform(-10, 10, 10)), label='Original Function')
plt.plot(new_individual, label='Optimized Individual')
plt.legend()
plt.show()