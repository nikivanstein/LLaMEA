import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import differential_evolution
from collections import deque

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

    def evolutionary_optimization(self, func, initial_population, mutation_rate, exploration_rate, mutation_function):
        # Initialize population with random functions
        population = initial_population

        # Run evolutionary optimization for specified number of iterations
        for _ in range(self.max_iter):
            # Select the best individual using hierarchical clustering
            cluster_labels = np.argpartition(func, self.current_dim)[-1]
            selected_individuals = [self.func[i] for i in range(len(self.func)) if cluster_labels == cluster_labels[self.current_dim]]

            # Perform mutation on selected individuals
            mutated_individuals = []
            for individual in selected_individuals:
                if np.random.rand() < mutation_rate:
                    mutated_individuals.append(mutation_function(individual))

            # Replace selected individuals with mutated ones
            population = [individual if i in selected_individuals else mutated_individuals[i] for i in range(len(population))]

            # Re-evaluate fitness of each individual
            fitness = [self.func(i) for i in population]
            population = [individual for _, individual in sorted(zip(fitness, population), reverse=True)]

            # Update current dimension
            self.current_dim += 1
            if self.budget == 0:
                break

        return population

# One-line description with main idea
# Hierarchical clustering with evolutionary optimization for efficient exploration-ejection in black box optimization problems

# Code