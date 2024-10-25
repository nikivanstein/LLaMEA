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

def select_best_individual(individual, func, budget, dim):
    if budget > 0:
        # Select the best individual using hierarchical clustering
        cluster_labels = np.argpartition(func, dim)[-1]
        if cluster_labels == dim:
            return individual
        else:
            # Select the best individual without hierarchical clustering
            return np.random.choice(individual, size=dim, replace=False)
    else:
        # No more individuals to evaluate
        return None

def mutate_individual(individual, func, mutation_rate):
    if random.random() < mutation_rate:
        # Perform mutation on the individual
        return individual + np.random.uniform(-5.0, 5.0, len(individual))
    else:
        # No mutation
        return individual

# Example usage:
noisy_black_box_optimizer = NoisyBlackBoxOptimizer(budget=100, dim=10)
func = lambda x: np.sin(x)
individual = np.random.uniform(-5.0, 5.0, 10)
best_individual = select_best_individual(individual, func, 100, 10)
best_individual = mutate_individual(best_individual, func, 0.1)
print("Best individual:", best_individual)
print("Best fitness:", np.mean([func(x) for x in best_individual]))