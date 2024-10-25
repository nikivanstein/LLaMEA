import numpy as np
import matplotlib.pyplot as plt
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

def evaluate_func(func, x, max_iter=1000):
    return minimize(func, x, method="SLSQP", bounds=[(-5.0, 5.0) for _ in range(func.shape[1])], options={"maxiter": max_iter})

# Example usage
optimizer = NoisyBlackBoxOptimizer(budget=100, dim=10)
func = lambda x: np.sin(x)
best_func = optimizer(func)
print(f"Best function: {best_func}")
print(f"Best fitness: {evaluate_func(best_func, np.array([0]))}")

# Refine the strategy
def refine_strategy(func, x, best_func):
    new_individual = func(x)
    updated_individual = evaluate_func(func, new_individual, max_iter=1000)
    return updated_individual, evaluate_func(updated_individual, x, max_iter=1000)

def plot_strategy(func, x, best_func):
    best_individual = best_func(x)
    updated_individual, fitness = refine_strategy(func, x, best_func)
    plot = plt.plot(x, best_individual, label="Best individual")
    plot.plot(x, updated_individual, label="Refined individual")
    plt.legend()
    plt.show()

plot_strategy(func, np.array([-1, 1]), best_func)