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
                    # Perform hierarchical clustering with gradient descent for efficient exploration-ejection
                    if self.current_dim == 0:
                        # Hierarchical clustering without gradient descent
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

def hierarchical_black_box(budget, dim, max_iter, initial_individual, initial_func):
    optimizer = NoisyBlackBoxOptimizer(budget, dim, max_iter)
    results = []
    for _ in range(max_iter):
        individual = initial_individual
        fitness = optimizer.func(initial_func)
        results.append((individual, fitness))
        # Refine the strategy by applying a probability of 0.023809523809523808 to change the individual lines of the selected solution
        if np.random.rand() < 0.023809523809523808:
            individual = np.random.choice(initial_individual.shape[0], p=initial_individual.shape[0] / initial_individual.shape[0] + (initial_individual.shape[0] - initial_individual.shape[0]) / (initial_individual.shape[0] + initial_individual.shape[0] * (1 - 0.023809523809523808)))
        individual = np.array([func(x) for func in individual])
        fitness = optimizer.func(individual)
        results.append((individual, fitness))
    return results

# Test the algorithm
initial_individual = np.array([-1.0, -1.0])
initial_func = lambda x: x**2
results = hierarchical_black_box(100, 10, 1000, initial_individual, initial_func)
for individual, fitness in results:
    print(f"Individual: {individual}, Fitness: {fitness}")