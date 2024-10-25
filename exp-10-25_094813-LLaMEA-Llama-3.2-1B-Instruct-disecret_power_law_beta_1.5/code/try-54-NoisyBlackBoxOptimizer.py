import numpy as np
import matplotlib.pyplot as plt
import random
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

    def mutate(self, individual):
        if random.random() < 0.01:
            idx = random.randint(0, self.dim - 1)
            self.func[individual[idx]] = np.random.uniform(-5.0, 5.0)
        return individual

    def evaluate_fitness(self, individual, budget):
        # Hierarchical clustering to refine the strategy
        cluster_labels = np.argpartition(self.func[individual], self.current_dim)[-1]
        # Perform gradient descent with hierarchical clustering for efficient exploration-ejection
        if self.explore_eviction:
            # Select the best function to optimize using hierarchical clustering
            best_individual = np.array([func(x) for x in np.random.uniform(-5.0, 5.0, self.dim) if cluster_labels == cluster_labels[self.current_dim]])
        else:
            # Perform gradient descent with hierarchical clustering for efficient exploration-ejection
            if self.current_dim == 0:
                # Gradient descent without hierarchical clustering
                best_individual = np.array([func(x) for x in np.random.uniform(-5.0, 5.0, self.dim)])
            else:
                # Hierarchical clustering to select the best function to optimize
                best_individual = np.array([func(x) for x in np.random.uniform(-5.0, 5.0, self.dim) if cluster_labels == cluster_labels[self.current_dim]])
            # Refine the strategy using hierarchical clustering
            self.func = np.array([func(x) for x in self.func])
            self.current_dim += 1
            if self.budget == 0:
                break
        # Update the best individual
        self.func = self.func[individual]
        return self.func

# Example usage:
optimizer = NoisyBlackBoxOptimizer(100, 10)
individual = random.randint(0, 9)
for _ in range(100):
    fitness = optimizer.func(individual)
    print(f"Individual: {individual}, Fitness: {fitness}")
    if random.random() < 0.1:
        individual = optimizer.mutate(individual)
    optimizer.evaluate_fitness(individual, 10)