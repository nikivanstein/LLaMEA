# Description: Hierarchical Black Box Optimization using Hierarchical Clustering and Genetic Algorithm
# Code: 
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import differential_evolution
from sklearn.cluster import KMeans
from typing import Dict, List

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
            # Hierarchical clustering to select the best function to optimize
            cluster_labels = np.argpartition(func, self.current_dim)[-1]
            self.func = np.array([func(x) for x in np.random.uniform(-5.0, 5.0, self.dim)])
            self.current_dim += 1
            if self.budget == 0:
                break
            # Genetic algorithm for efficient exploration-ejection
            self.population = [self.func]
            self.fitness = [self.func]
            for _ in range(self.max_iter):
                new_population = []
                new_fitness = []
                for i in range(len(self.population)):
                    for j in range(self.dim):
                        new_individual = self.func.copy()
                        new_individual[j] += np.random.normal(0, 1)
                        new_population.append(new_individual)
                        new_fitness.append(self.func.copy())
                new_population = np.array(new_population)
                new_fitness = np.array(new_fitness)
                # Hierarchical clustering to select the best individual to optimize
                cluster_labels = np.argpartition(new_fitness, self.current_dim)[-1]
                self.func = new_population[cluster_labels]
                self.population = new_population
                self.fitness = new_fitness
                if self.budget == 0:
                    break
            return self.func

    def func(self, x):
        return np.array([func(x) for func in self.func])

# One-line description with main idea
# Hierarchical Black Box Optimization using Hierarchical Clustering and Genetic Algorithm
# to solve black box optimization problems