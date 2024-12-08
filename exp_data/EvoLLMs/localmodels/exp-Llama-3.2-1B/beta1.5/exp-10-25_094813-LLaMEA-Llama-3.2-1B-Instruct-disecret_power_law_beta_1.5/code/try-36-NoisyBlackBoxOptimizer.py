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
            # Hierarchical clustering with gradient descent for efficient exploration-ejection
            while self.budget > 0 and self.current_dim < self.dim:
                if self.explore_eviction:
                    # Select the best function to optimize using hierarchical clustering
                    cluster_labels = np.argpartition(func, self.current_dim)[-1]
                    self.explore_eviction = False
                else:
                    # Perform hierarchical clustering with gradient descent
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

def plot_history(func, initial, final, budget):
    x = np.linspace(-5.0, 5.0, 100)
    y = func(x)
    plt.plot(x, y)
    plt.scatter(initial, y[initial], label='Initial')
    plt.scatter(final, y[final], label='Final')
    plt.legend()
    plt.show()

# Test the optimizer
noisy_black_box_optimizer = NoisyBlackBoxOptimizer(budget=1000, dim=10)
initial_individual = np.array([-1, -1, -1, -1, -1, -1, -1, -1, -1, -1])
final_individual = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1])

plot_history(func=noisy_black_box_optimizer.func, initial=initial_individual, final=final_individual, budget=1000)